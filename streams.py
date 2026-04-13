import os
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import numpy.typing as npt
import pandas as pd
from ezpadova import parsec
from gala.coordinates import GreatCircleICRSFrame
from galstreams import MWStreams

ISOCHRONES_DIR = "./isochrones"
MWS = MWStreams()


@dataclass
class StreamConfig:
    """
    Configuration for extracting a stellar stream from Gaia.
    """

    name: str
    metallicity: float  # [Fe/H]
    age_yr: float  # years
    pm_phi1_range: tuple[float, float]  # mas/yr
    pm_phi2_range: tuple[float, float]  # mas/yr
    g_mag_range: tuple[float, float] = (14, 21)
    ruwe_max: float = 1.4
    parallax_max: float = 1.0  # mas
    notes: str = ""

    def __post_init__(self):
        self._galstreams_properties()
        self.isochrone = self._get_isochrone()
        self.cmd_polygon = self._build_cmd_polygon()

    def _galstreams_properties(self) -> None:
        track_names = MWS.get_track_names_for_stream(self.name)
        for track_name in track_names:
            try:
                track = MWS[track_name]
                self.notes += f" Galstreams discovery: {track.ref_discovery}."
                break
            except KeyError:
                pass
        self.stream_frame: GreatCircleICRSFrame = track.stream_frame
        track = track.track
        track_phi = track.transform_to(self.stream_frame)
        self.phi1_range: tuple[float, float] = (
            track_phi.phi1.min().deg - 5,
            track_phi.phi1.max().deg + 5,
        )
        self.phi2_range: tuple[float, float] = (
            track_phi.phi2.min().deg - 2.5,
            track_phi.phi2.max().deg + 2.5,
        )
        self.ra_range: tuple[float, float] = (
            track.ra.min().deg - 5,
            track.ra.max().deg + 5,
        )
        self.dec_range: tuple[float, float] = (
            track.dec.min().deg - 2.5,
            track.dec.max().deg + 2.5,
        )
        self.distance_kpc: float = float(np.median(track.distance.kpc))

    @property
    def distance_modulus(self) -> float:
        return 5 * np.log10(self.distance_kpc * 1000) - 5

    def _get_isochrone(self) -> pd.DataFrame:
        """
        Query an isochrone for this stream's distance and metallicity.
        """
        if os.path.exists(f"{ISOCHRONES_DIR}/{self.name}.csv"):
            return pd.read_csv(f"{ISOCHRONES_DIR}/{self.name}.csv")

        isochrone: pd.DataFrame = parsec.get_isochrones(
            age_yr=(self.age_yr, self.age_yr, 0),
            MH=(self.metallicity, self.metallicity, 0),
            photsys_file="gaiaEDR3",
        )
        isochrone["Gmag"] += self.distance_modulus
        isochrone["G_BPmag"] += self.distance_modulus
        isochrone["G_RPmag"] += self.distance_modulus
        isochrone["BP_RP"] = isochrone["G_BPmag"] - isochrone["G_RPmag"]
        isochrone = isochrone[
            isochrone["label"].isin([1, 2, 3])
        ]  # keep only main sequence, subgiant, and red giant branch
        isochrone.to_csv(f"{ISOCHRONES_DIR}/{self.name}.csv", index=False)
        return isochrone

    def _build_cmd_polygon(self, color_buffer: float = 0.15) -> npt.NDArray:
        """
        Build a CMD selection polygon from a PARSEC isochrone.
        """
        main_sequence = self.isochrone[self.isochrone["label"] == 1].copy()
        # Blue edge, then red edge reversed, close
        poly = np.column_stack(
            [
                np.concatenate(
                    [
                        main_sequence["BP_RP"] - color_buffer,
                        (main_sequence["BP_RP"] + color_buffer)[::-1],
                        [main_sequence["BP_RP"].iloc[0] - color_buffer],
                    ]
                ),
                np.concatenate(
                    [
                        main_sequence["Gmag"],
                        main_sequence["Gmag"][::-1],
                        [main_sequence["Gmag"].iloc[0]],
                    ]
                ),
            ]
        )
        return poly


class Streams:
    """
    Pre-defined stream configurations based on literature measurements.
    See notes in StreamConfig for references.
    """

    GD1 = StreamConfig(
        name="GD-1",
        metallicity=-2.2,
        age_yr=12e9,
        pm_phi1_range=(-15.0, -7.5),
        pm_phi2_range=(-6.0, 0.0),
        notes="Koposov+2010; Price-Whelan & Bonaca 2018",
    )

    PAL5 = StreamConfig(
        name="Pal5",
        metallicity=-1.4,
        age_yr=12e9,
        pm_phi1_range=(-3.5, -1.0),
        pm_phi2_range=(-1.5, 1.0),
        notes="Erkal+2017; Price-Whelan+2019; Bonaca+2020",
    )

    PHLEGETHON = StreamConfig(
        name="Phlegethon",
        metallicity=-1.4,
        age_yr=10e9,
        pm_phi1_range=(-45.0, -25.0),
        pm_phi2_range=(-5.0, 0),
        notes="Ibata+2018",
    )

    SYLGR = StreamConfig(
        name="Sylgr",
        metallicity=-2.9,
        age_yr=12e9,
        pm_phi1_range=(-1.0, 1.0),
        pm_phi2_range=(-1.0, 1.0),
        notes="Ibata+2019b; Roederer & Gnedin 2019",
    )

    JHELUM = StreamConfig(
        name="Jhelum",
        metallicity=-1.8,
        age_yr=12e9,
        pm_phi1_range=(-10.0, -5.0),
        pm_phi2_range=(0, 6.5),
        notes="",
    )
