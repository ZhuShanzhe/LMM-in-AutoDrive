"""Export a lightweight top-down SVG preview of the generated main-road map."""

from pathlib import Path


ROAD_LENGTH_M = 5000.0
LANE_WIDTH_M = 3.5
LANES_PER_DIRECTION = 3
SHOULDER_WIDTH_M = 1.0
SIDEWALK_WIDTH_M = 2.0
INTERSECTION_S = 1200.0

SCALE_X = 0.16
SCALE_Y = 8.0
MARGIN_X = 80
MARGIN_Y = 80
WIDTH = int(ROAD_LENGTH_M * SCALE_X + MARGIN_X * 2)
HEIGHT = 420
CENTER_Y = HEIGHT / 2


def sx(s):
    return MARGIN_X + s * SCALE_X


def sy(t):
    return CENTER_Y - t * SCALE_Y


def line(x1, y1, x2, y2, cls, extra=""):
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}" {extra}/>'


def rect(x, y, w, h, cls, extra=""):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" class="{cls}" {extra}/>'


def text(x, y, value, cls):
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}">{value}</text>'


def export_svg():
    half_drive = LANES_PER_DIRECTION * LANE_WIDTH_M
    road_edge = half_drive + SHOULDER_WIDTH_M
    sidewalk_edge = road_edge + SIDEWALK_WIDTH_M

    road_top = sy(road_edge)
    road_bottom = sy(-road_edge)
    road_height = road_bottom - road_top
    side_top = sy(sidewalk_edge)
    side_bottom = sy(-sidewalk_edge)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<style>",
        "svg{background:#222;font-family:Arial,sans-serif}",
        ".asphalt{fill:#303030}",
        ".sidewalk{fill:#686862}",
        ".building{fill:#8f969b;stroke:#b7bdc1;stroke-width:1}",
        ".building2{fill:#747b80;stroke:#a7adb1;stroke-width:1}",
        ".lane{stroke:#d8d8cf;stroke-width:1.2;stroke-dasharray:9 13}",
        ".edge{stroke:#c8c8bf;stroke-width:1.4}",
        ".yellow{stroke:#e4bd43;stroke-width:1.5}",
        ".crosswalk{stroke:#e6e6dc;stroke-width:2}",
        ".stop{stroke:#eeeeea;stroke-width:3}",
        ".curb{stroke:#8b8b86;stroke-width:2}",
        ".label{fill:#f0f0e8;font-size:15px}",
        ".small{fill:#d0d0c8;font-size:11px}",
        "</style>",
        rect(sx(0), side_top, ROAD_LENGTH_M * SCALE_X, side_bottom - side_top, "sidewalk"),
        rect(sx(0), road_top, ROAD_LENGTH_M * SCALE_X, road_height, "asphalt"),
    ]

    for t in (-road_edge, road_edge):
        parts.append(line(sx(0), sy(t), sx(ROAD_LENGTH_M), sy(t), "curb"))

    for t in (-half_drive, half_drive):
        parts.append(line(sx(0), sy(t), sx(ROAD_LENGTH_M), sy(t), "edge"))

    parts.append(line(sx(0), sy(-0.18), sx(ROAD_LENGTH_M), sy(-0.18), "yellow"))
    parts.append(line(sx(0), sy(0.18), sx(ROAD_LENGTH_M), sy(0.18), "yellow"))

    for t in (-LANE_WIDTH_M, -2 * LANE_WIDTH_M, LANE_WIDTH_M, 2 * LANE_WIDTH_M):
        parts.append(line(sx(0), sy(t), sx(ROAD_LENGTH_M), sy(t), "lane"))

    # Visual intersection at 1200 m, while the main route remains straight.
    ix = sx(INTERSECTION_S)
    parts.append(rect(ix - 18, side_top, 36, side_bottom - side_top, "asphalt", 'opacity="0.95"'))
    for offset in (-10, -6, -2, 2, 6, 10):
        parts.append(line(ix + offset, sy(-half_drive), ix + offset, sy(half_drive), "crosswalk"))
    parts.append(line(ix - 24, sy(-half_drive), ix - 24, sy(0), "stop"))
    parts.append(line(ix - 24, sy(0), ix - 24, sy(half_drive), "stop"))

    # Dense city blocks on both sides, two rows per side.
    block_id = 0
    for s in range(120, int(ROAD_LENGTH_M), 180):
        for side in (-1, 1):
            for row, setback in enumerate((7.0, 14.0)):
                t = side * (sidewalk_edge + setback)
                for offset, length, depth in ((-55, 26, 7), (-18, 32, 8), (24, 28, 7), (58, 22, 7)):
                    x = sx(max(0, min(ROAD_LENGTH_M, s + offset))) - length * SCALE_X / 2
                    y = sy(t) - depth * SCALE_Y / 2
                    cls = "building" if (block_id + row) % 2 == 0 else "building2"
                    parts.append(rect(x, y, length * SCALE_X, depth * SCALE_Y, cls))
                    block_id += 1

    for s in range(120, int(ROAD_LENGTH_M), 120):
        for t in (sidewalk_edge + 0.8, -(sidewalk_edge + 0.8)):
            parts.append(f'<circle cx="{sx(s):.1f}" cy="{sy(t):.1f}" r="2.0" fill="#d6cf9c"/>')

    parts.extend([
        text(28, 34, "VLA_MainRoad_5km - top-down preview", "label"),
        text(28, 54, "Straight 5 km urban arterial, bidirectional 6 lanes, visual intersection at 1200 m", "small"),
        text(ix - 30, side_top - 14, "1200 m intersection", "small"),
        "</svg>",
    ])
    return "\n".join(parts)


def main():
    path = Path(__file__).resolve().parent / "maps" / "output" / "VLA_MainRoad_5km_preview.svg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(export_svg(), encoding="utf-8")
    print("Generated:", path)


if __name__ == "__main__":
    main()
