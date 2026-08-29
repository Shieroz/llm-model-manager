import json, random
random.seed(11)
nouns=["sensor","relay","conduit","lattice","aperture","manifold","gyroscope","capacitor","filament","resonator"]
N=6500
lines=[f"Log entry {i:05d}: the {random.choice(nouns)} in sector {random.randint(100,999)} "
       f"reported a drift of {random.uniform(0,10):.3f} units during cycle {random.randint(1,50)}."
       for i in range(N)]

# (depth, text) -- inserted deepest-first so indices stay valid
facts=[
 # 1. three-hop arithmetic chain, spread far apart
 (0.05,"Register A7 was initialised to the value 3391."),
 (0.44,"The contents of register A7 were copied verbatim into register K2."),
 (0.81,"Register K2 was then doubled and the result written to register M9."),
 # 2. distractor set -- four near-identical lines, only one is the Meridian array
 (0.12,"The calibration constant for the Tessellate array is 51.884 kelvin-seconds."),
 (0.29,"The calibration constant for the Meridian array is 74.219 kelvin-seconds."),
 (0.53,"The calibration constant for the Halcyon array is 63.507 kelvin-seconds."),
 (0.72,"The calibration constant for the Peridot array is 88.132 kelvin-seconds."),
 # 3. exact long alphanumeric ID
 (0.37,"Backup resonator serial number 7QX-40277-BJ19 is stored in vault B, shelf 12."),
 # 4. supersession chain -- must return the LAST value, not the first
 (0.18,"The emergency purge threshold was set to 12.5 bar."),
 (0.61,"The emergency purge threshold was revised from 12.5 bar down to 9.75 bar."),
 (0.90,"Following the audit, the emergency purge threshold was revised once more to 8.25 bar."),
 # 5. two-point comparison requiring exact retrieval of both
 (0.23,"Special calibration run: the aperture in sector 404 reported a drift of 2.118 units."),
 (0.67,"Special calibration run: the aperture in sector 811 reported a drift of 6.902 units."),
]
for d,t in sorted(facts, key=lambda x:-x[0]):
    lines.insert(int(len(lines)*d), t)

q=("\n\nUsing ONLY the log above, answer all six. One line each, formatted exactly as "
   "`<number>. <answer>` with no explanation:\n"
   "1. What value ended up in register M9?\n"
   "2. What is the calibration constant for the Meridian array?\n"
   "3. What is the full serial number of the backup resonator?\n"
   "4. What is the current emergency purge threshold?\n"
   "5. Which of the two special calibration run sectors reported the higher drift?\n"
   "6. What drift value did the aperture in sector 404 report?")
prompt="Below is a long maintenance log.\n\n"+"\n".join(lines)+q
json.dump({"model":"Qwen3.8-27B-NoVision-UD-Q4_K_M","messages":[{"role":"user","content":prompt}],
  "max_tokens":400,"stream":False,"temperature":0.0,
  "chat_template_kwargs":{"enable_thinking":False}}, open('/tmp/claude-1000/-home-llm/221f090b-d729-45a9-8704-578fa8f3e79c/scratchpad/eval.json','w'))
print("built, chars:",len(prompt))
