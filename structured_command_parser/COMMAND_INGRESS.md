# Command Ingress Contract

This contract joins an external ASR plus translation component to the checked-in
ModernBERT parser. It does not implement speech recognition, translation, or
vehicle control.

## Input Snapshot

The upstream component writes one complete UTF-8 JSON object and atomically
renames it into place. The parser service accepts only translated English:

```json
{
  "request_id": "voice-0001",
  "text": "Slow down and stop before the red truck.",
  "language": "en-US",
  "modality": "VOICE"
}
```

`request_id` is mandatory and must identify one user utterance. `language` must
be `en`, `en-US`, or `en-GB`; Chinese text is intentionally rejected here so a
missing translation stage cannot silently reach the English parser.

## Service Boundary

`run_english_parser_service.py` warms the parser once, ignores byte-identical
input snapshots, and atomically writes two files:

- `driving_intent.json`: schema-valid `DrivingIntent 1.1.0` for the scene and
  decision modules.
- `parser_receipt.json`: request ID, modality, parser method, status, and the
  parser-only latency.

Example interface-only invocation:

```bash
python -m structured_command_parser.scripts.run_english_parser_service \
  --input runtime/translated_command.json \
  --driving-intent-output runtime/driving_intent.json \
  --receipt-output runtime/parser_receipt.json \
  --once
```

The parser-only latency in the receipt is not an end-to-end voice latency. The
final evaluation must separately measure ASR, translation, scene processing,
planning, controller application, and any queued-frame delay.

## Safety and License Boundary

The current model is permitted for offline parsing and interface validation
only. Do not connect this service's output to a CARLA vehicle controller unless
the model/data license has been replaced or explicit permission for that use is
obtained. The CARLA control boundary still defaults to a safe stop for missing,
invalid, future, or stale decisions.
