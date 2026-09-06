"""Interactive 3D Gaussian Splatting & Radiance Field Laboratory Workbench.

Demonstrates real-time 3D radiance field volume rendering in the terminal using
sub-pixel Unicode Braille graphics, spherical harmonics specular lobes, and
orbital camera turntable animation.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

from chromasplat.projection import Camera
from chromasplat.renderer import GaussianRenderer
from chromasplat.scene import SceneFactory
from chromasplat.visualizer import SplatVisualizer


def run_workbench() -> None:
    """Parse command line arguments and execute interactive Gaussian Splatting laboratory."""
    parser = argparse.ArgumentParser(
        description="ChromaSplat: Interactive 3D Gaussian Splatting & Radiance Field Workbench"
    )
    parser.add_argument(
        "--scene",
        choices=["rings", "cornell", "dna"],
        default="rings",
        help="Procedural scene to render (default: rings)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=64,
        help="Terminal canvas character width (default: 64)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=22,
        help="Terminal canvas character height (default: 22)",
    )
    parser.add_argument(
        "--sh-degree",
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help="Maximum spherical harmonics degree for directional color (default: 2)",
    )
    parser.add_argument(
        "--azimuth",
        type=float,
        default=35.0,
        help="Initial camera orbit azimuth in degrees (default: 35.0)",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=18.0,
        help="Initial camera orbit elevation in degrees (default: 18.0)",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=3.4,
        help="Camera orbit distance in meters (default: 3.4)",
    )
    parser.add_argument(
        "--orbit-frames",
        type=int,
        default=1,
        help="Number of turntable orbit frames to animate (default: 1)",
    )
    parser.add_argument(
        "--save-ppm",
        type=str,
        default="",
        help="Optional file path to export rendered frame as PPM image",
    )

    args = parser.parse_args()

    # 1. Instantiate procedural scene
    if args.scene == "rings":
        scene = SceneFactory.orbiting_rings(num_planet_splats=70, num_ring_splats=180)
        default_dist = 3.4
        default_el = 18.0
    elif args.scene == "cornell":
        scene = SceneFactory.cornell_box()
        default_dist = 3.8
        default_el = 10.0
    elif args.scene == "dna":
        scene = SceneFactory.dna_double_helix(num_turns=2.5, total_height=3.2, radius=0.8)
        default_dist = 4.2
        default_el = 15.0
    else:
        scene = SceneFactory.orbiting_rings()
        default_dist = 3.4
        default_el = 18.0

    cam_dist = args.distance if args.distance != 3.4 else default_dist
    cam_el = args.elevation if args.elevation != 18.0 else default_el

    # 2. Setup visualizer and rendering pipeline
    visualizer = SplatVisualizer(char_width=args.width, char_height=args.height)
    renderer = GaussianRenderer(sh_degree=args.sh_degree, tile_size=16)

    min_b, max_b = scene.get_bounds()
    print("=" * visualizer.canvas.char_width)
    print(" CHROMASPLAT: 3D GAUSSIAN SPLATTING LABORATORY WORKBENCH")
    print(f" Scene: {scene.name:<18} Primitives: {len(scene)} Gaussians")
    print(f" Bounds: X[{min_b[0]:.2f}, {max_b[0]:.2f}] Y[{min_b[1]:.2f}, {max_b[1]:.2f}] Z[{min_b[2]:.2f}, {max_b[2]:.2f}]")
    print("=" * visualizer.canvas.char_width)

    # 3. Render frame(s)
    num_frames = max(1, args.orbit_frames)
    for frame_idx in range(num_frames):
        if num_frames > 1:
            current_az = args.azimuth + (360.0 / num_frames) * frame_idx
        else:
            current_az = args.azimuth

        cam = Camera.orbit(
            width=visualizer.pixel_width,
            height=visualizer.pixel_height,
            azimuth_deg=current_az,
            elevation_deg=cam_el,
            distance=cam_dist,
            fov_y_deg=60.0,
        )

        res = renderer.render(scene, cam)
        hud_output = visualizer.render_frame_with_hud(
            scene=scene,
            camera=cam,
            render_result=res,
            profile=renderer.last_profile,
            azimuth_deg=current_az,
            elevation_deg=cam_el,
            distance=cam_dist,
        )

        # Clear screen escape for animated turntable
        if num_frames > 1:
            sys.stdout.write("\033[H")
        sys.stdout.write(hud_output + "\n")
        sys.stdout.flush()

        # Save PPM if requested on first frame
        if frame_idx == 0 and args.save_ppm:
            with open(args.save_ppm, "wb") as f_out:
                f_out.write(res.to_ppm_bytes())
            print(f"Exported PPM image to: {args.save_ppm}")

        if num_frames > 1:
            time.sleep(0.04)  # ~25 FPS animation pacing


if __name__ == "__main__":
    run_workbench()
