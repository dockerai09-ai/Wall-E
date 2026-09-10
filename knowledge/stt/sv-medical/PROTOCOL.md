# Claude Pro arm — capture protocol

Claude Pro has no speech-to-text API: audio cannot be uploaded to the Claude
API, and the claude.ai app's voice mode is a consumer dictation feature.
That arm therefore cannot be run by a script. It is captured by hand, once,
and imported as a JSONL file so both arms are scored by exactly the same code.

1. Synthesize the clips: `./synth.sh` (writes `audio/<id>.wav`).
2. On a phone with the Claude app signed in to a Pro account, start a new
   chat, open voice mode (or the microphone dictation button) and set the
   phone next to the computer's speaker.
3. Play each clip (`afplay audio/c01.wav`), let Claude transcribe it, and
   copy the text it produced *before* any reply. If Claude answers instead of
   showing the transcription, ask it first: "Skriv ner exakt vad jag säger,
   ordagrant, utan att svara."
4. Put one line per clip in `arms/claude-pro.jsonl`:
   `{"id":"c01","hypothesis":"...","source":"claude.ai voice mode","captured_at":"YYYY-MM-DD"}`
   (`arms/claude-pro.template.jsonl` shows the shape). Clips Claude could not
   transcribe get `"hypothesis":""` — an empty transcript scores as fully
   wrong, which is the honest outcome.
5. Import and score: `npm run stt:bench -- --import knowledge/stt/sv-medical/arms/claude-pro.jsonl --arm claude-pro`
6. The loop reads that run as its target: `npm run stt:loop`.

Record what the app actually did (dictation language setting, whether
Swedish was offered, whether it refused clinical content). Those
observations belong in the comparison next to the numbers.
