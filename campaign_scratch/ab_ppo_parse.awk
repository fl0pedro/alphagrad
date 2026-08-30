# gawk -f ab_ppo_parse.awk ppo_TAG.log
# Emits one TSV row per host_log call (warm-up rollouts included, flagged).
# cols: idx  kind  sec_ep  host_total  <phase>=<sec> ...
/\[health ep[0-9]+\]/ {
  if (match($0, /\[health ep[0-9]+\] ppo=([^ ]+).*sec\/ep=([0-9.]+)/, m)) {
    hi++; hppo[hi] = m[1]; hsec[hi] = m[2]
  }
}
/\[prof ep=/ {
  if (match($0, /\[prof ep=[ 0-9]+\] host_total=[ ]*([0-9.]+)s(.*)/, m)) {
    pi++; ptot[pi] = m[1]; prest[pi] = m[2]
  }
}
END {
  n = (hi < pi ? hi : pi)
  for (i = 1; i <= n; i++) {
    kind = (hppo[i] == "nan" ? "warmup" : "episode")
    printf "%d\t%s\tsec_ep=%s\thost_total=%s", i, kind, hsec[i], ptot[i]
    r = prest[i]
    while (match(r, /(cb\.[a-z_0-9]+|faces\.[a-z_0-9]+|prof\/[a-z_0-9]+)=([0-9.]+)s/, q)) {
      printf "\t%s=%s", q[1], q[2]
      r = substr(r, RSTART + RLENGTH)
    }
    printf "\n"
  }
}
