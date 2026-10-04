"""Command-line interface: `facerec <command> ...` or `python -m facerec <command> ...`."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .images import draw_faces, iter_images, read_image, write_image


def _recognizer(args, threshold=True):
    from .recognizer import FaceRecognizer

    return FaceRecognizer(
        db=args.db,
        model_dir=args.model_dir,
        det_size=args.det_size,
        threshold=args.threshold if threshold else None,
        margin=args.margin if threshold else None,
        providers=["CPUExecutionProvider"] if args.cpu else None,
    )


def _people_dirs(root: Path) -> list[Path]:
    dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not dirs:
        sys.exit(f"{root} has no sub-folders. Expected one folder per person, e.g. {root}/Alice/*.jpg")
    return dirs


def _enroll_paths(rec, name: str, paths, force: bool) -> tuple[int, int]:
    added = total = 0
    for path in iter_images(paths):
        total += 1
        img = read_image(path)
        if img is None:
            print(f"  [skip] {path}: unreadable image")
            continue
        result = rec.enroll(name, img, source=str(path), force=force)
        added += result.added
        print(f"  [{'+' if result.added else 'skip'}] {path.name}: {result.message}")
    return added, total


def cmd_enroll(args) -> None:
    rec = _recognizer(args)
    added, total = _enroll_paths(rec, args.name, args.paths, args.force)
    if total == 0:
        sys.exit("no images found")
    rec.gallery.save()
    n = rec.gallery.identities().get(args.name, 0)
    print(f"{args.name}: added {added}/{total} photos, {n} samples stored.")
    if n < 3:
        print("Tip: 3-10 varied photos per person (angles, lighting, glasses) noticeably improves accuracy.")


def cmd_enroll_dir(args) -> None:
    rec = _recognizer(args)
    for person in _people_dirs(Path(args.root)):
        print(f"{person.name}:")
        added, total = _enroll_paths(rec, person.name, [person], args.force)
        print(f"  -> {added}/{total} added")
    rec.gallery.save()
    print(f"Gallery now has {len(rec.gallery.identities())} people.")


def cmd_list(args) -> None:
    from .gallery import Gallery

    g = Gallery(args.db)
    ids = g.identities()
    if not ids:
        print(f"No one enrolled in {args.db} yet.")
        return
    for name, count in ids.items():
        print(f"{name:30s} {count} sample{'s' if count != 1 else ''}")
    print(f"\nthreshold={g.threshold:.2f} margin={g.margin:.2f}")


def cmd_remove(args) -> None:
    from .gallery import Gallery

    g = Gallery(args.db)
    removed = g.remove(args.name)
    if not removed:
        sys.exit(f"'{args.name}' is not enrolled")
    g.save()
    print(f"Removed {args.name} ({removed} samples).")


def cmd_recognize(args) -> None:
    rec = _recognizer(args)
    if not len(rec.gallery):
        print("Warning: nobody is enrolled yet, every face will be Unknown.", file=sys.stderr)
    found = False
    for path in iter_images(args.paths):
        found = True
        img = read_image(path)
        if img is None:
            print(f"{path}: unreadable image")
            continue
        faces = rec.identify(img)
        print(f"{path}: {len(faces)} face(s)")
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int)
            m = f.match
            detail = f"score {m.score:.2f}"
            if not m.known and m.best_name:
                why = "; ".join(f.quality.issues) or (
                    "below threshold" if m.score < rec.gallery.threshold else f"too close to {m.runner_up}"
                )
                detail += f", closest: {m.best_name} ({why})"
            print(f"  {f.name:20s} {detail}  box=({x1},{y1},{x2},{y2})")
        if args.output:
            out = Path(args.output)
            dest = out / f"{path.stem}_faces{path.suffix}" if out.suffix == "" else out
            write_image(dest, draw_faces(img, faces))
        if args.show:
            import cv2

            cv2.imshow("facerec", draw_faces(img, faces))
            if cv2.waitKey(0) & 0xFF in (ord("q"), 27):
                break
    if not found:
        sys.exit("no images found")


def cmd_verify(args) -> None:
    rec = _recognizer(args)
    a, b = read_image(args.image_a), read_image(args.image_b)
    if a is None or b is None:
        sys.exit("could not read both images")
    sim, err = rec.verify(a, b)
    if sim is None:
        sys.exit(err)
    same = sim >= rec.gallery.threshold
    print(f"similarity {sim:.3f} -> {'SAME person' if same else 'DIFFERENT people'} "
          f"(threshold {rec.gallery.threshold:.2f})")


def cmd_video(args) -> None:
    import cv2

    from .tracker import FaceTracker

    rec = _recognizer(args)
    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"could not open video source {args.source}")
    writer = None
    tracker = FaceTracker()
    fps, last = 0.0, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            faces = rec.identify_frame(frame, tracker)
            vis = draw_faces(frame, faces)
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now
            cv2.putText(vis, f"{fps:.1f} fps", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2, cv2.LINE_AA)
            if args.output:
                if writer is None:
                    h, w = vis.shape[:2]
                    rate = cap.get(cv2.CAP_PROP_FPS) or 25
                    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), rate, (w, h))
                writer.write(vis)
            if not args.no_display:
                cv2.imshow("facerec (q to quit)", vis)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()


def cmd_calibrate(args) -> None:
    from .evaluate import evaluate

    rec = _recognizer(args, threshold=False)
    embs, labels = [], []
    for person in _people_dirs(Path(args.root)):
        for path in iter_images([person]):
            img = read_image(path)
            faces = rec.analyze(img, max_faces=1) if img is not None else []
            if not faces:
                print(f"  [skip] {path}: no face")
                continue
            embs.append(faces[0].embedding)
            labels.append(person.name)
    if len(set(labels)) < 2:
        sys.exit("need photos of at least 2 people (2+ photos each) to calibrate")

    report = evaluate(np.stack(embs), labels, rec.gallery.threshold, rec.gallery.margin)
    g, i = report.genuine, report.impostor
    print(f"\n{report.num_people} people, {report.num_images} photos")
    if g.size:
        print(f"same-person similarity:      mean {g.mean():.2f}, lowest {g.min():.2f}")
    print(f"different-person similarity: mean {i.mean():.2f}, highest {i.max():.2f}")
    print(f"closed-set rank-1 accuracy:  {report.rank1:.1%}\n")
    print("                         known named right  strangers rejected  misidentified")
    for label, p in (("current", report.current), ("recommended", report.best)):
        print(f"{label:12s} t={p.threshold:.2f} m={p.margin:.2f}   {p.known_accuracy:7.1%}"
              f"            {p.unknown_rejection:7.1%}          {p.misidentified:6.1%}")
    if args.save:
        rec.gallery.config.update(threshold=report.best.threshold, margin=report.best.margin)
        rec.gallery.save()
        print(f"\nSaved threshold={report.best.threshold:.2f} margin={report.best.margin:.2f} to {args.db}")
    else:
        print("\nRun again with --save to use the recommended settings.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="facerec", description="Accurate face recognition (SCRFD + ArcFace).")
    p.add_argument("--db", default="face_db", help="face database folder (default: ./face_db)")
    p.add_argument("--model-dir", help="where the ONNX models live (default: ~/.facerec/models/buffalo_l)")
    p.add_argument("--det-size", type=int, default=640,
                   help="detector input size; raise to 960+ for tiny faces in big photos")
    p.add_argument("--threshold", type=float, help="override the match threshold (cosine similarity)")
    p.add_argument("--margin", type=float, help="override the required lead over the runner-up")
    p.add_argument("--cpu", action="store_true", help="force CPU even if a GPU provider is installed")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("enroll", help="add photos of one person")
    s.add_argument("name")
    s.add_argument("paths", nargs="+", help="image files and/or folders")
    s.add_argument("--force", action="store_true", help="skip quality and consistency checks")
    s.set_defaults(func=cmd_enroll)

    s = sub.add_parser("enroll-dir", help="enroll everyone from ROOT/<person name>/*.jpg")
    s.add_argument("root")
    s.add_argument("--force", action="store_true", help="skip quality and consistency checks")
    s.set_defaults(func=cmd_enroll_dir)

    s = sub.add_parser("list", help="show enrolled people")
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("remove", help="delete a person")
    s.add_argument("name")
    s.set_defaults(func=cmd_remove)

    s = sub.add_parser("recognize", help="identify faces in images")
    s.add_argument("paths", nargs="+", help="image files and/or folders")
    s.add_argument("-o", "--output", help="save annotated image (file) or images (folder)")
    s.add_argument("--show", action="store_true", help="display results in a window")
    s.set_defaults(func=cmd_recognize)

    s = sub.add_parser("verify", help="are these two photos the same person?")
    s.add_argument("image_a")
    s.add_argument("image_b")
    s.set_defaults(func=cmd_verify)

    s = sub.add_parser("video", help="live recognition from a webcam or video file")
    s.add_argument("source", nargs="?", default="0", help="camera index or video path (default: 0)")
    s.add_argument("-o", "--output", help="save annotated video (.mp4)")
    s.add_argument("--no-display", action="store_true", help="don't open a window")
    s.set_defaults(func=cmd_video)

    s = sub.add_parser("calibrate", help="measure accuracy on labelled photos and tune the threshold")
    s.add_argument("root", help="folder with one sub-folder of photos per person")
    s.add_argument("--save", action="store_true", help="store the recommended threshold in the database")
    s.set_defaults(func=cmd_calibrate)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
