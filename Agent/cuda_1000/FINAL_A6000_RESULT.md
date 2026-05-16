# Final A6000 result

Target GPU only: `NVIDIA RTX A6000`.

Best valid observed A6000 run:

```text
NVIDIA RTX A6000, driver 550.127.08
[cuda-half-perf] prompt=29 gen=100 decode_only=0.110762s tok/s=902.837 graph=1
```

Final extra attempt requested by user:

- Tried changing only the logits partial-argmax kernel from 8 warps/block to 16 warps/block.
- It produced `900.257 tok/s` but changed output tokens, so it is invalid and was not kept.

Conclusion: did not reach 1000 TPS on A6000. Best valid A6000 result remains `902.837 tok/s`.

Invalid non-A6000 note: RTX PRO 6000 Blackwell reached >1000 TPS earlier, but that is not counted for this A6000 target.
