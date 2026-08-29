import re,sys
t=sys.stdin.read()
keys=[("1","6782"),("2","74.219"),("3","7QX-40277-BJ19"),("4","8.25"),("5","811"),("6","2.118")]
score=0
for n,exp in keys:
    m=re.search(rf"^\s*{n}[.):]\s*(.+)$", t, re.M)
    got=(m.group(1).strip() if m else "<missing>")
    ok=exp.lower() in got.lower(); score+=ok
    print(f"  Q{n}: {'PASS' if ok else 'FAIL'}  expected={exp!r}  got={got[:60]!r}")
print(f"  SCORE: {score}/6")
