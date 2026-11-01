#!/usr/bin/env python3
"""Offline, unqualified layer-3 TP4/rank-0 slices for a paging experiment."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import struct


def slice_i16(data, shape, axis, start, stop):
    if (not shape or any(type(n) is not int or n <= 0 for n in shape)
            or axis not in (0, 1) or axis >= len(shape)
            or not 0 <= start < stop <= shape[axis]
            or len(data) != math.prod(shape) * 2):
        raise ValueError('invalid shape, bounds or short I16 tensor')
    stride = math.prod(shape[axis + 1:]) * 2
    group = shape[axis] * stride
    return b''.join(data[base + start*stride:base + stop*stride]
                    for base in range(0, len(data), group))


def validate_spec(spec, dtype, shape):
    if spec.get('dtype') != dtype or spec.get('shape') != shape:
        raise ValueError(f'expected {dtype} {shape}, found {spec}')
    offsets = spec.get('data_offsets')
    size = math.prod(shape) * {'I16': 2, 'F16': 2, 'I32': 4}[dtype]
    if (not isinstance(offsets, list) or len(offsets) != 2
            or any(type(x) is not int for x in offsets)
            or offsets[0] < 0 or offsets[1] - offsets[0] != size):
        raise ValueError(f'invalid tensor extent: {spec}')
    return offsets


def self_test():
    shape = [3, 4, 2]
    values = [1000*a + 100*b + c for a in range(3) for b in range(4) for c in range(2)]
    raw = struct.pack('<24h', *values)
    expected0 = [1000*a + 100*b + c for a in range(1, 3) for b in range(4) for c in range(2)]
    expected1 = [1000*a + 100*b + c for a in range(3) for b in range(1, 3) for c in range(2)]
    assert slice_i16(raw, shape, 0, 1, 3) == struct.pack('<16h', *expected0)
    assert slice_i16(raw, shape, 1, 1, 3) == struct.pack('<12h', *expected1)
    bad_calls = [
        lambda: slice_i16(raw[:-1], shape, 1, 0, 1),
        lambda: slice_i16(raw, [3, 5, 2], 0, 0, 1),
        lambda: slice_i16(raw, shape, 1, 0, 5),
        lambda: validate_spec({'dtype': 'I16', 'shape': [3], 'data_offsets': [0, 6]}, 'I16', [4]),
        lambda: validate_spec({'dtype': 'I16', 'shape': [3], 'data_offsets': [0, 5]}, 'I16', [3]),
    ]
    for call in bad_calls:
        try:
            call()
        except ValueError:
            continue
        raise AssertionError('invalid input was accepted')
    print('PASS: nonuniform axis0/axis1 slices; short input, wrong shape, bounds and extent rejected', flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_shard(path):
    with path.open('rb') as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError(f'short safetensors prefix: {path.name}')
        length, = struct.unpack('<Q', prefix)
        if not 2 <= length <= 64 * 1024 * 1024:
            raise ValueError(f'invalid header length: {path.name}')
        raw = source.read(length)
        if len(raw) != length:
            raise ValueError(f'short safetensors header: {path.name}')
    header = json.loads(raw)
    ranges = []
    for name, spec in header.items():
        if name == '__metadata__':
            continue
        offsets = spec.get('data_offsets')
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(x) is not int for x in offsets)
                or not 0 <= offsets[0] <= offsets[1]):
            raise ValueError(f'invalid offsets: {name}')
        ranges.append(tuple(offsets))
    end = 0
    for start, stop in sorted(ranges):
        if start != end:
            raise ValueError(f'gap or overlap in {path.name}')
        end = stop
    if path.stat().st_size != 8 + length + end:
        raise ValueError(f'shard file size mismatch: {path.name}')
    return header, 8 + length


def extract(source, output):
    source = source.resolve(strict=True)
    output = output.absolute()
    repo = Path(__file__).resolve().parents[2]
    if output.resolve().is_relative_to(repo):
        raise ValueError('raw evidence must be outside the source checkout')
    index_path = source / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text())['weight_map']
    prefix = 'model.language_model.layers.3.mlp.experts.'
    projections = ('gate_proj', 'up_proj', 'down_proj')
    shards = {}
    for expert in range(288):
        for projection in projections:
            for kind in ('trellis', 'suh', 'svh', 'mcg'):
                key = f'{prefix}{expert}.{projection}.{kind}'
                name = index[key]
                if Path(name).name != name:
                    raise ValueError('source shard name must be a basename')
                if name not in shards:
                    shards[name] = load_shard(source / name)

    def tensor(expert, projection, kind, dtype, shape):
        key = f'{prefix}{expert}.{projection}.{kind}'
        name = index[key]
        header, base = shards[name]
        start, stop = validate_spec(header[key], dtype, shape)
        with (source / name).open('rb') as handle:
            handle.seek(base + start)
            data = handle.read(stop - start)
        if len(data) != stop - start:
            raise ValueError(f'short tensor read: {key}')
        if dtype == 'F16' and any(value & 0x7c00 == 0x7c00 for value, in struct.iter_unpack('<H', data)):
            raise ValueError(f'nonfinite transform: {key}')
        return data

    # Verify every required tensor, including the fixed MCG seed, before writing outputs.
    for expert in range(288):
        for projection in projections:
            down = projection == 'down_proj'
            specs = (
                ('trellis', 'I16', [128, 256, 64] if down else [256, 128, 64]),
                ('suh', 'F16', [2048] if down else [4096]),
                ('svh', 'F16', [4096] if down else [2048]),
                ('mcg', 'I32', [1]),
            )
            for kind, dtype, shape in specs:
                key = f'{prefix}{expert}.{projection}.{kind}'
                validate_spec(shards[index[key]][0][key], dtype, shape)
            if tensor(expert, projection, 'mcg', 'I32', [1]) != bytes.fromhex('ed1faccb'):
                raise ValueError(f'unexpected MCG seed: expert {expert} {projection}')

    output.mkdir(parents=True, exist_ok=False)
    print(f'Extracting unqualified slices to {output}', flush=True)
    trellises = output / 'trellises.raw.partial'
    with trellises.open('xb') as target:
        for expert in range(288):
            for projection in projections:
                down = projection == 'down_proj'
                shape = [128, 256, 64] if down else [256, 128, 64]
                raw = tensor(expert, projection, 'trellis', 'I16', shape)
                # Adjacent little-endian I16 values already have the required I32 byte layout.
                target.write(slice_i16(raw, shape, 0 if down else 1, 0, 32))
        target.flush()
        os.fsync(target.fileno())
    transforms = output / 'transforms.raw.partial'
    layout = [('suh', 'gate_proj', 4096, 4096), ('suh', 'up_proj', 4096, 4096),
              ('suh', 'down_proj', 2048, 512), ('svh', 'gate_proj', 2048, 512),
              ('svh', 'up_proj', 2048, 512), ('svh', 'down_proj', 4096, 4096)]
    with transforms.open('xb') as target:
        for kind, projection, original, keep in layout:
            for expert in range(288):
                target.write(tensor(expert, projection, kind, 'F16', [original])[:keep*2])
        target.flush()
        os.fsync(target.fileno())
    outputs = {}
    for path, expected in ((trellises, 288 * 3 * 1024 * 1024), (transforms, 7962624)):
        if path.stat().st_size != expected:
            raise ValueError(f'output size mismatch: {path.name}')
        final = path.with_suffix('')
        path.rename(final)
        outputs[final.name] = {'bytes': expected, 'sha256': sha256(final)}
    files = {}
    for name in ['model.safetensors.index.json', *sorted(shards)]:
        path = source / name
        files[name] = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
    manifest = {
        'experiment': 'betelgeuse-paging-component-slice',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_snapshot': str(source), 'source_revision': source.name,
        'source_files': files, 'extractor_sha256': sha256(Path(__file__)),
        'layer': 3, 'expert_order': '0..287', 'projections': list(projections),
        'slice': {'tp_size': 4, 'rank': 0, 'gate_up_trellis': 'axis1[0:32]',
                  'down_trellis': 'axis0[0:32]', 'trellis_encoding': 'little-endian I16 bytes reinterpreted as adjacent I32',
                  'mcg_hex_bytes': 'ed1faccb',
                  'transforms': [{'kind': k, 'projection': p, 'source_elements': n, 'keep_first': m}
                                 for k, p, n, m in layout]},
        'qualification': {'qualified_tp_sizes': [], 'serving_reader_qualified': False,
                          'status': 'UNQUALIFIED TP4 slicing',
                          'limitation': 'Paging parity compares the same supplied slices; it does not establish a correct TP4 model, inference or model quality.'},
        'outputs': outputs,
    }
    temporary = output / 'manifest.json.partial'
    with temporary.open('x') as handle:
        json.dump(manifest, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.rename(output / 'manifest.json')
    print(json.dumps({'output': str(output), 'outputs': outputs, 'manifest': str(output / 'manifest.json')}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, help='fresh evidence directory outside Git')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.source is None or args.output is None:
        parser.error('--source and --output are required unless --self-test is set')
    extract(args.source, args.output)


if __name__ == '__main__':
    main()
