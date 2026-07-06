from __future__ import annotations

from pathlib import Path


def write_hbond_csv(
    path: Path,
    counts: dict[str, dict[str, int]],
    fractions: dict[str, dict[str, float]],
    samples_total: dict[str, int],
    classes: dict[str, list[str]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("species,hbond_class,count,total_oxygen_samples,fraction\n")
        for species in counts:
            for class_label in classes[species]:
                handle.write(
                    f"{species},{class_label},{counts[species][class_label]},"
                    f"{samples_total[species]},{fractions[species][class_label]:.10g}\n"
                )


def write_raw_hbond_csv(
    path: Path,
    raw_counts: dict[str, dict[str, int]],
    grouped_counts: dict[str, dict[str, int]],
    samples_total: dict[str, int],
    classes: dict[str, list[str]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("species,raw_hbond_class,grouped_hbond_class,count,total_oxygen_samples,fraction\n")
        for species in raw_counts:
            for raw_label, count in sorted(raw_counts[species].items(), key=lambda item: (-item[1], item[0])):
                grouped_label = raw_label if raw_label in classes[species] else "other"
                if grouped_label not in grouped_counts[species]:
                    grouped_label = ""
                fraction = count / samples_total[species] if samples_total[species] else 0.0
                handle.write(
                    f"{species},{raw_label},{grouped_label},{count},"
                    f"{samples_total[species]},{fraction:.10g}\n"
                )
