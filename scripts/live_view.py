"""Step 0: just LOOK at the signals. Opens a window with load / current / position
error for every joint, scrolling live.

    # no arm needed: simulated arm with random "hand pushes"
    python scripts/live_view.py --fake

    # real arm, torque off (move it by hand, see what the signals do when limp)
    python scripts/live_view.py

    # real arm HOLDING its current pose at 30 % torque. Now push it with your hand.
    python scripts/live_view.py --hold arm --torque-limit 300

Every run writes data/live_view_<stamp>.csv (new file each time).
"""
from _common import RateMeter, base_parser, open_and_report, parse_joints

from tactile import config
from tactile.liveplot import Live, Panel, run_headless
from tactile.logger import CsvLogger

p = base_parser(__doc__)
p.add_argument("--hold", default="none",
               help="joints to hold in place: none | arm | all | comma list (e.g. elbow_flex,wrist_flex)")
p.add_argument("--torque-limit", type=int, default=300, help="Torque_Limit for held joints, 0..1000 (default 300 = 30%%)")
p.add_argument("--joints", default="all", help="which joints to plot")
args = p.parse_args()

hold = parse_joints(args.hold)
if args.fake and not hold:
    hold = parse_joints("arm")   # a limp fake arm shows flat lines; hold it so the pushes show up
show = parse_joints(args.joints)
src = open_and_report(args)
log = CsvLogger("live_view", config.JOINTS, fake=args.fake)

panels = [
    Panel("Present_Load (drive duty, -1000..1000)", [f"load/{j}" for j in show]),
    Panel("Present_Current (raw, ~6.5 mA each, unverified)", [f"cur/{j}" for j in show]),
    Panel("position error = goal - present (held joints only, ticks)", [f"err/{j}" for j in show]),
]
live = Live("live_view: SO-101 torque signals", panels)


def worker(lv: Live):
    if hold:
        src.hold_here(hold, args.torque_limit)
        print(f"[tactile] holding {hold} at Torque_Limit {args.torque_limit}. Push the arm!")
    rate = RateMeter()
    while not lv.stop.is_set():
        s = src.read()
        log.write(s)
        vals = {}
        for j in show:
            vals[f"load/{j}"] = s.load[j]
            vals[f"cur/{j}"] = s.cur[j]
            vals[f"err/{j}"] = s.err[j]
        lv.push(s.t, vals)
        lv.hz = rate.tick()
        bad = {j: v for j, v in s.status.items() if v}
        if bad:
            lv.set_banner(f"STATUS ERROR {bad} (overload / heat / voltage?)", "red")
        else:
            lv.set_banner(f"holding {', '.join(hold)}" if hold else "torque untouched (limp?)", "black")


try:
    if args.no_window:
        run_headless(worker, args.duration)
    else:
        live.run(worker, args.duration)
finally:
    log.close()
    src.close(release=args.release_on_exit)
