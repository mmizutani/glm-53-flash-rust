//! Synthetic, one-layer, one-rank EXL3 paging experiment. No engine dependencies.
#[cfg(not(test))]
use std::path::Path;
use std::{
    fs::{File, OpenOptions},
    io::{self, Write},
    os::unix::fs::FileExt,
    time::Instant,
};
const EXPERTS: usize = 288;
const CAPACITY: usize = 8;
const MATRIX: usize = 1 << 20;
const PAYLOAD: usize = 3 * MATRIX;
const HEADER: usize = 4096;
const RECORD: usize = HEADER + PAYLOAD;
fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
fn hash(bytes: &[u8]) -> u64 {
    use std::hash::Hasher;
    // Internal fresh-run fixture format; record the Rust version alongside source hashes.
    let mut h = std::collections::hash_map::DefaultHasher::new();
    h.write(bytes);
    h.finish()
}
fn encode(id: usize, bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER + bytes.len());
    out.extend_from_slice(b"BTPAGE01");
    out.extend_from_slice(&(id as u32).to_le_bytes());
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&hash(bytes).to_le_bytes());
    out.resize(HEADER, 0);
    out.extend_from_slice(bytes);
    out
}
fn decode(record: &[u8], id: usize, size: usize) -> io::Result<&[u8]> {
    if record.len() != HEADER + size || &record[..8] != b"BTPAGE01" {
        return Err(invalid("record length/magic"));
    }
    let u32_at = |at| u32::from_le_bytes(record[at..at + 4].try_into().unwrap()) as usize;
    let expected = u64::from_le_bytes(record[16..24].try_into().unwrap());
    if u32_at(8) != id || u32_at(12) != size || hash(&record[HEADER..]) != expected {
        return Err(invalid("record identity/length/checksum"));
    }
    Ok(&record[HEADER..])
}
fn record_offset(id: usize) -> io::Result<u64> {
    if id >= EXPERTS {
        return Err(invalid("expert ID out of range"));
    }
    (id as u64)
        .checked_mul(RECORD as u64)
        .ok_or_else(|| invalid("record offset overflow"))
}
fn read_record_timed(file: &File, id: usize, record: &mut [u8]) -> io::Result<[f64; 2]> {
    let offset = record_offset(id)?;
    if record.len() != RECORD {
        return Err(invalid("staging record length"));
    }
    let start = Instant::now();
    file.read_exact_at(record, offset)?;
    let read_ms = start.elapsed().as_secs_f64() * 1000.;
    let start = Instant::now();
    decode(record, id, PAYLOAD)?;
    Ok([read_ms, start.elapsed().as_secs_f64() * 1000.])
}
#[cfg(not(test))]
fn read_record(file: &File, id: usize) -> io::Result<Vec<u8>> {
    let mut record = vec![0; RECORD];
    read_record_timed(file, id, &mut record)?;
    Ok(record)
}
enum Staging {
    Allocating(Vec<Vec<u8>>),
    Reused(Vec<u8>),
    #[cfg(not(test))]
    Pinned(*mut u8),
}
impl Staging {
    fn new(mode: &str) -> io::Result<Self> {
        match mode {
            "allocating" => Ok(Self::Allocating(Vec::new())),
            "reused" => Ok(Self::Reused(vec![0; CAPACITY * RECORD])),
            #[cfg(not(test))]
            "pinned" => {
                let ptr = unsafe { probe_host_alloc(CAPACITY * RECORD) };
                if ptr.is_null() {
                    return Err(invalid("null pinned allocation"));
                }
                // Rust byte slices require initialized storage, before any request reads it.
                unsafe {
                    std::ptr::write_bytes(ptr, 0, CAPACITY * RECORD);
                }
                Ok(Self::Pinned(ptr))
            }
            _ => Err(invalid("staging mode must be allocating, reused or pinned")),
        }
    }
    fn prepare(&mut self, count: usize) -> io::Result<()> {
        if count > CAPACITY {
            return Err(invalid("staging capacity exceeded"));
        }
        if let Self::Allocating(records) = self {
            records.clear();
            records.extend((0..count).map(|_| vec![0; RECORD]));
        }
        Ok(())
    }
    fn record_mut(&mut self, i: usize) -> &mut [u8] {
        assert!(i < CAPACITY);
        match self {
            Self::Allocating(records) => &mut records[i],
            Self::Reused(bytes) => &mut bytes[i * RECORD..(i + 1) * RECORD],
            #[cfg(not(test))]
            Self::Pinned(ptr) => unsafe {
                std::slice::from_raw_parts_mut(ptr.add(i * RECORD), RECORD)
            },
        }
    }
}
impl Drop for Staging {
    fn drop(&mut self) {
        #[cfg(not(test))]
        if let Self::Pinned(ptr) = self {
            unsafe {
                probe_host_free(*ptr);
            }
        }
    }
}
#[derive(Clone, Debug, PartialEq)]
struct Pager {
    slots: Vec<Option<usize>>,
}
impl Pager {
    fn new(capacity: usize) -> Self {
        Self {
            slots: vec![None; capacity],
        }
    }
    fn plan(&self, active: &[usize]) -> io::Result<(Self, Vec<(usize, usize)>, Vec<i32>)> {
        if active.len() > self.slots.len() || active.iter().any(|id| *id >= EXPERTS) {
            return Err(invalid("active union exceeds capacity or has invalid ID"));
        }
        for (i, id) in active.iter().enumerate() {
            if active[..i].contains(id) {
                return Err(invalid("duplicate active ID"));
            }
        }
        let mut next = self.clone();
        let mut missing = Vec::new();
        let mut ids = Vec::new();
        for &id in active {
            let slot = if let Some(slot) = next.slots.iter().position(|v| *v == Some(id)) {
                slot
            } else {
                let slot = next
                    .slots
                    .iter()
                    .position(|v| v.is_none_or(|old| !active.contains(&old)))
                    .ok_or_else(|| invalid("no evictable slot"))?;
                next.slots[slot] = Some(id);
                missing.push((slot, id));
                slot
            };
            ids.push(slot as i32);
        }
        Ok((next, missing, ids))
    }
    fn fill(
        &mut self,
        active: &[usize],
        file: &File,
        staging: &mut Staging,
        mut upload: impl FnMut(usize, &[u8]),
    ) -> io::Result<(Vec<i32>, usize, [f64; 3])> {
        let (next, missing, ids) = self.plan(active)?;
        // Validate every missing record before the first GPU write; mapping commits after upload.
        staging.prepare(missing.len())?;
        let mut timing = [0f64; 3];
        for (i, &(_, id)) in missing.iter().enumerate() {
            let t = read_record_timed(file, id, staging.record_mut(i))?;
            timing[0] += t[0];
            timing[1] += t[1];
        }
        let start = Instant::now();
        for (i, (slot, _)) in missing.iter().enumerate() {
            upload(*slot, &staging.record_mut(i)[HEADER..]);
        }
        timing[2] = start.elapsed().as_secs_f64() * 1000.;
        if let Staging::Allocating(records) = staging {
            records.clear();
        }
        *self = next;
        Ok((ids, missing.len(), timing))
    }
}
#[cfg(not(test))]
unsafe extern "C" {
    fn probe_host_alloc(bytes: usize) -> *mut u8;
    fn probe_host_free(ptr: *mut u8);
    fn posix_fadvise(fd: i32, offset: i64, length: i64, advice: i32) -> i32;
    fn probe_init() -> *mut std::ffi::c_void;
    fn probe_transforms(s: *mut std::ffi::c_void, bytes: *const u8);
    fn probe_fill(s: *mut std::ffi::c_void, pooled: i32, slot: i32, bytes: *const u8);
    fn probe_run(
        s: *mut std::ffi::c_void,
        pooled: i32,
        graph: i32,
        picks: *const i32,
        ids: *const i32,
        output: *mut f32,
    ) -> f32;
    fn probe_destroy(s: *mut std::ffi::c_void);
}
#[cfg(not(test))]
fn make_pack(path: &Path, raw: Option<&File>) -> io::Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut bytes = vec![0u8; PAYLOAD];
    for id in 0..EXPERTS {
        if let Some(raw) = raw {
            raw.read_exact_at(&mut bytes, (id * PAYLOAD) as u64)?;
        } else {
            let mut rng = (id as u64 + 1).wrapping_mul(0x9e3779b97f4a7c15);
            for chunk in bytes.chunks_exact_mut(8) {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                chunk.copy_from_slice(&rng.to_le_bytes());
            }
        }
        file.write_all(&encode(id, &bytes))?;
    }
    file.sync_all()
}
#[cfg(not(test))]
fn io_read_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/io")
        .unwrap()
        .lines()
        .find_map(|l| l.strip_prefix("read_bytes: ").map(|v| v.parse().unwrap()))
        .unwrap()
}
#[cfg(not(test))]
fn percentile(values: &[f64], p: f64) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    ordered[((ordered.len() as f64 * p).ceil() as usize).saturating_sub(1)]
}
#[cfg(not(test))]
fn main() -> io::Result<()> {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: probe OUTSIDE_GIT_EVIDENCE_DIRECTORY");
    let model_dir = std::env::args().nth(2).filter(|v| v != "-");
    let staging_mode = std::env::args().nth(3).unwrap_or_else(|| "reused".into());
    let raw = if let Some(model) = &model_dir {
        let raw = File::open(Path::new(model).join("trellises.raw"))?;
        if raw.metadata()?.len() != (EXPERTS * PAYLOAD) as u64 {
            return Err(invalid("raw trellis length"));
        }
        Some(raw)
    } else {
        None
    };
    std::fs::create_dir_all(&dir)?;
    let path = Path::new(&dir).join(if raw.is_some() {
        "model-experts.pack"
    } else {
        "synthetic-experts.pack"
    });
    // Always create a fresh pack, so a reused evidence directory cannot select stale source data.
    make_pack(&path, raw.as_ref())?;
    let file = File::open(path)?;
    if file.metadata()?.len() != (RECORD * EXPERTS) as u64 {
        return Err(invalid("pack total length"));
    }
    println!(
        "checkpoint_derived={} experts={EXPERTS} pool_capacity={CAPACITY} matrix_bytes={MATRIX} record_bytes={RECORD}",
        model_dir.is_some()
    );
    println!(
        "resident_trellis_bytes={} pool_trellis_bytes={} resident_transform_bytes={} reference_coexists=true",
        EXPERTS * PAYLOAD,
        CAPACITY * PAYLOAD,
        EXPERTS * 3 * (4096 + 512) * 2
    );
    let transforms = if let Some(model) = &model_dir {
        let bytes = std::fs::read(Path::new(model).join("transforms.raw"))?;
        if bytes.len() != EXPERTS * 3 * (4096 + 512) * 2 {
            return Err(invalid("transform length"));
        }
        Some(bytes)
    } else {
        None
    };
    let state = unsafe { probe_init() };
    if let Some(bytes) = &transforms {
        unsafe {
            probe_transforms(state, bytes.as_ptr());
        }
    }
    for id in 0..EXPERTS {
        let record = read_record(&file, id)?;
        unsafe {
            probe_fill(state, 0, id as i32, record[HEADER..].as_ptr());
        }
    }
    let mut sets = vec![
        vec![287, 0, 73, 144, 215, 36, 108, 251],
        vec![215, 1, 287, 89, 36, 177, 0, 266],
        vec![2, 39, 76, 113, 150, 187, 224, 261],
    ];
    sets.extend(
        (0..EXPERTS)
            .step_by(8)
            .map(|start| (start..start + 8).collect::<Vec<_>>()),
    );
    let picks: Vec<Vec<i32>> = sets
        .iter()
        .map(|set| set.iter().map(|v| *v as i32).chain([288]).collect())
        .collect();
    let mut references = Vec::new();
    for p in &picks {
        let mut out = vec![0f32; 9 * 4096];
        unsafe {
            probe_run(state, 0, 0, p.as_ptr(), p.as_ptr(), out.as_mut_ptr());
        }
        if !out.iter().all(|v| v.is_finite()) {
            return Err(invalid("nonfinite reference"));
        }
        references.push(out);
    }
    let mut staging = Staging::new(&staging_mode)?;
    println!(
        "staging_mode={staging_mode} fixed_staging_bytes={}",
        if staging_mode == "allocating" {
            0
        } else {
            CAPACITY * RECORD
        }
    );
    let mut pager = Pager::new(CAPACITY);
    let (ids, _, _) = pager.fill(&sets[0], &file, &mut staging, |slot, b| unsafe {
        probe_fill(state, 1, slot as i32, b.as_ptr())
    })?;
    let mut wrong = ids.clone();
    wrong.rotate_left(1);
    wrong.push(288);
    let mut output = vec![0f32; 9 * 4096];
    unsafe {
        probe_run(
            state,
            1,
            0,
            picks[0].as_ptr(),
            wrong.as_ptr(),
            output.as_mut_ptr(),
        );
    }
    let different = output[..8 * 4096]
        .iter()
        .zip(&references[0])
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    if different == 0 {
        return Err(invalid(
            "wrong-map negative control failed to change output",
        ));
    }
    println!(
        "wrong_map_negative_control_changed={different}/{}",
        8 * 4096
    );
    println!(
        "mode,scenario,samples,host_p50_ms,host_p95_ms,host_p99_ms,gpu_p50_ms,gpu_p95_ms,gpu_p99_ms,fill_p50_ms,fill_p95_ms,fill_p99_ms,misses,logical_payload_bytes,proc_read_bytes,parity_values,read_p50_ms,validate_p50_ms,upload_p50_ms"
    );
    for graph in [0, 1] {
        for scenario in [
            "resident",
            "all_hit",
            "overlap_4_miss",
            "full_8_miss",
            "disjoint_advised",
        ] {
            pager = Pager::new(CAPACITY);
            pager.fill(&sets[0], &file, &mut staging, |slot, b| unsafe {
                probe_fill(state, 1, slot as i32, b.as_ptr())
            })?;
            let mut hosts = Vec::new();
            let mut gpus = Vec::new();
            let mut fills = Vec::new();
            let mut reads = Vec::new();
            let mut verifies = Vec::new();
            let mut uploads = Vec::new();
            let mut misses = 0;
            let mut parity = 0;
            let advised = scenario == "disjoint_advised";
            if advised {
                pager = Pager::new(CAPACITY);
                use std::os::fd::AsRawFd;
                let result = unsafe { posix_fadvise(file.as_raw_fd(), 0, 0, 4) };
                if result != 0 {
                    return Err(io::Error::from_raw_os_error(result));
                }
            }
            let warmups = if advised { 0 } else { 5 };
            let samples = if advised { EXPERTS / 8 } else { 200 };
            let mut before = io_read_bytes();
            for rep in 0..warmups + samples {
                if rep == warmups {
                    before = io_read_bytes();
                }
                let selected = match scenario {
                    "disjoint_advised" => 3 + rep,
                    "overlap_4_miss" => {
                        if rep % 2 == 0 {
                            1
                        } else {
                            0
                        }
                    }
                    "full_8_miss" => {
                        if rep % 2 == 0 {
                            2
                        } else {
                            0
                        }
                    }
                    _ => 0,
                };
                let begin = Instant::now();
                let fill_begin = Instant::now();
                let (mut ids, count, timing) = if scenario == "resident" {
                    (
                        sets[selected].iter().map(|v| *v as i32).collect(),
                        0,
                        [0.; 3],
                    )
                } else {
                    pager.fill(&sets[selected], &file, &mut staging, |slot, b| unsafe {
                        probe_fill(state, 1, slot as i32, b.as_ptr())
                    })?
                };
                ids.push(288);
                let fill_ms = fill_begin.elapsed().as_secs_f64() * 1000.;
                let gpu = unsafe {
                    probe_run(
                        state,
                        (scenario != "resident") as i32,
                        graph,
                        picks[selected].as_ptr(),
                        ids.as_ptr(),
                        output.as_mut_ptr(),
                    )
                };
                let host = begin.elapsed().as_secs_f64() * 1000.;
                if output
                    .iter()
                    .zip(&references[selected])
                    .any(|(a, b)| !a.is_finite() || a.to_bits() != b.to_bits())
                {
                    return Err(invalid("exact parity failure"));
                }
                if rep >= warmups {
                    hosts.push(host);
                    gpus.push(gpu as f64);
                    fills.push(fill_ms);
                    reads.push(timing[0]);
                    verifies.push(timing[1]);
                    uploads.push(timing[2]);
                    misses += count;
                    parity += 8 * 4096;
                }
            }
            println!(
                "{},{scenario},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{misses},{},{},{},{:.6},{:.6},{:.6}",
                if graph == 1 { "graph" } else { "eager" },
                hosts.len(),
                percentile(&hosts, 0.5),
                percentile(&hosts, 0.95),
                percentile(&hosts, 0.99),
                percentile(&gpus, 0.5),
                percentile(&gpus, 0.95),
                percentile(&gpus, 0.99),
                percentile(&fills, 0.5),
                percentile(&fills, 0.95),
                percentile(&fills, 0.99),
                misses * PAYLOAD,
                io_read_bytes() - before,
                parity,
                percentile(&reads, 0.5),
                percentile(&verifies, 0.5),
                percentile(&uploads, 0.5)
            );
        }
    }
    unsafe {
        probe_destroy(state);
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn staging_reuses_storage_and_preserves_payloads() {
        let path = std::env::temp_dir().join(format!("betelgeuse-staging-{}", std::process::id()));
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        for id in 0..2 {
            file.write_all(&encode(id, &vec![id as u8; PAYLOAD]))
                .unwrap();
        }
        for mode in ["allocating", "reused"] {
            let mut staging = Staging::new(mode).unwrap();
            assert!(staging.prepare(CAPACITY + 1).is_err());
            staging.prepare(2).unwrap();
            let initial = staging.record_mut(0).as_ptr();
            for active in [[0, 1], [1, 0]] {
                let mut pager = Pager::new(CAPACITY);
                let mut payloads = Vec::new();
                let (_, misses, _) = pager
                    .fill(&active, &file, &mut staging, |_, b| {
                        payloads.push((b[0], b[PAYLOAD - 1]))
                    })
                    .unwrap();
                assert_eq!(misses, 2);
                assert_eq!(payloads, active.map(|v| (v as u8, v as u8)));
                if let Staging::Allocating(records) = &staging {
                    assert!(
                        records.is_empty(),
                        "allocating baseline must release per-request payloads"
                    );
                }
                if mode == "reused" {
                    assert_eq!(initial, staging.record_mut(0).as_ptr());
                }
            }
        }
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn protects_entire_active_union() {
        let pager = Pager {
            slots: vec![Some(0), Some(1), Some(2), Some(3)],
        };
        let (next, missing, ids) = pager.plan(&[4, 0, 5, 2]).unwrap();
        assert_eq!(missing, vec![(1, 4), (3, 5)]);
        assert_eq!(ids, vec![1, 0, 3, 2]);
        assert_eq!(next.slots, vec![Some(0), Some(4), Some(2), Some(5)]);
        assert_eq!(pager.slots, vec![Some(0), Some(1), Some(2), Some(3)]);
    }
    #[test]
    fn rejects_overflow_invalid_and_duplicates() {
        let pager = Pager::new(2);
        assert!(pager.plan(&[0, 1, 2]).is_err());
        assert!(pager.plan(&[288]).is_err());
        assert!(pager.plan(&[0, 0]).is_err());
        assert!(record_offset(usize::MAX).is_err());
        assert!(record_offset(287).is_ok());
    }
    #[test]
    fn rejects_short_corrupt_and_wrong_identity() {
        let good = encode(7, b"abcdefgh");
        assert_eq!(decode(&good, 7, 8).unwrap(), b"abcdefgh");
        assert!(decode(&good[..good.len() - 1], 7, 8).is_err());
        let mut corrupt = good.clone();
        corrupt[HEADER + 2] ^= 1;
        assert!(decode(&corrupt, 7, 8).is_err());
        assert!(decode(&good, 8, 8).is_err());
    }
    #[test]
    fn corrupt_second_record_never_uploads_or_commits() {
        let path = std::env::temp_dir().join(format!("betelgeuse-corrupt-{}", std::process::id()));
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        let payload = vec![0u8; PAYLOAD];
        file.write_all(&encode(0, &payload)).unwrap();
        let mut corrupt = encode(1, &payload);
        corrupt[HEADER + 17] ^= 1;
        file.write_all(&corrupt).unwrap();
        for mode in ["allocating", "reused"] {
            let mut pager = Pager::new(CAPACITY);
            let before = pager.clone();
            let mut uploads = 0;
            assert!(
                pager
                    .fill(&[0, 1], &file, &mut Staging::new(mode).unwrap(), |_, _| {
                        uploads += 1
                    })
                    .is_err()
            );
            assert_eq!(uploads, 0);
            assert_eq!(pager, before);
        }
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn truncated_fill_never_uploads_or_commits() {
        let path =
            std::env::temp_dir().join(format!("betelgeuse-truncated-{}", std::process::id()));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        file.set_len(17).unwrap();
        for mode in ["allocating", "reused"] {
            let mut pager = Pager::new(CAPACITY);
            let before = pager.clone();
            let mut uploads = 0;
            assert!(
                pager
                    .fill(&[0], &file, &mut Staging::new(mode).unwrap(), |_, _| {
                        uploads += 1
                    })
                    .is_err()
            );
            assert_eq!(uploads, 0);
            assert_eq!(pager, before);
        }
        std::fs::remove_file(path).unwrap();
    }
}
