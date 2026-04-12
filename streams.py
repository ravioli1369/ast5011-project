import os
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd
from ezpadova import parsec

ISOCHRONES_DIR = "./isochrones"


@dataclass
class StreamConfig:
    """
    Configuration for extracting a stellar stream from Gaia.
    """

    name: str
    origin_ra: float  # deg
    origin_dec: float  # deg
    R: np.ndarray  # 3×3 rotation matrix (ICRS → stream)
    phi1_range: tuple[float, float]  # deg
    phi2_range: tuple[float, float]  # deg
    distance_kpc: float
    metallicity: float  # [Fe/H]
    age_yr: float  # years
    pm_phi1_range: tuple[float, float]  # mas/yr
    pm_phi2_range: tuple[float, float]  # mas/yr
    g_mag_range: tuple[float, float] = (14, 21)
    ruwe_max: float = 1.4
    parallax_max: float = 1.0  # mas
    notes: str = ""
    isochrone: pd.DataFrame = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """
        Query an isochrone for this stream's distance and metallicity.
        """
        if os.path.exists(f"{ISOCHRONES_DIR}/{self.name}.csv"):
            self.isochrone = pd.read_csv(f"{ISOCHRONES_DIR}/{self.name}.csv")
            return

        isochrone = parsec.get_isochrones(
            age_yr=(self.age_yr, self.age_yr, 0),
            MH=(self.metallicity, self.metallicity, 0),
            photsys_file="gaiaEDR3",
        )
        isochrone["Gmag"] += self.distance_modulus
        isochrone["G_BPmag"] += self.distance_modulus
        isochrone["G_RPmag"] += self.distance_modulus
        isochrone["BP_RP"] = isochrone["G_BPmag"] - isochrone["G_RPmag"]
        self.isochrone = pd.DataFrame(isochrone)
        self.isochrone.to_csv(f"{ISOCHRONES_DIR}/{self.name}.csv", index=False)

    @property
    def R_inv(self) -> np.ndarray:
        """Inverse rotation (stream → ICRS)."""
        return self.R.T

    @property
    def distance_modulus(self) -> float:
        return 5 * np.log10(self.distance_kpc * 1000) - 5

    @property
    def ra_range(self) -> tuple[float, float]:
        ra, _ = self._icrs_bounds
        return (float(np.floor(ra.min()) - 1), float(np.ceil(ra.max()) + 1))

    @property
    def dec_range(self) -> tuple[float, float]:
        _, dec = self._icrs_bounds
        return (
            float(max(-90, np.floor(dec.min()) - 1)),
            float(min(90, np.ceil(dec.max()) + 1)),
        )

    @cached_property
    def _icrs_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Inverse-transform the stream rectangle to RA/Dec.
        """
        phi1_grid, phi2_grid = np.meshgrid(
            np.linspace(self.phi1_range[0], self.phi1_range[1], 500),
            np.linspace(self.phi2_range[0], self.phi2_range[1], 500),
        )
        p1 = np.deg2rad(phi1_grid.ravel())
        p2 = np.deg2rad(phi2_grid.ravel())

        xyz_stream = np.vstack(
            [
                np.cos(p2) * np.cos(p1),
                np.cos(p2) * np.sin(p1),
                np.sin(p2),
            ]
        )
        xyz_icrs = self.R_inv @ xyz_stream

        ra = np.rad2deg(np.arctan2(xyz_icrs[1], xyz_icrs[0])) % 360
        dec = np.rad2deg(np.arcsin(np.clip(xyz_icrs[2], -1, 1)))
        return ra, dec


class Streams:
    """
    Pre-defined stream configurations based on literature measurements.
    See notes in StreamConfig for references.
    """

    GD1 = StreamConfig(
        name="GD-1",
        origin_ra=147.185,
        origin_dec=33.402,
        R=np.array(
            [
                [-0.4776303088, -0.1738432154, 0.8611897727],
                [0.510844589, -0.8524449229, 0.111245042],
                [0.7147776536, 0.4930681392, 0.4959603976],
            ]
        ),
        phi1_range=(-100, 20),
        phi2_range=(-8, 5),
        distance_kpc=7.5,
        metallicity=-2.2,
        age_yr=12e9,
        pm_phi1_range=(-15.0, -7.5),
        pm_phi2_range=(-6.0, 0.0),
        notes="Koposov+2010; Price-Whelan & Bonaca 2018",
    )

    PAL5 = StreamConfig(
        name="Palomar 5",
        origin_ra=231.977,
        origin_dec=1.243,
        R=np.array(
            [
                [-0.6546801354, -0.1100875638, 0.7478470685],
                [-0.0890787883, -0.9775498082, -0.1910780062],
                [0.7505782736, -0.1777799025, 0.6363368810],
            ]
        ),
        phi1_range=(-20, 15),
        phi2_range=(-5, 5),
        distance_kpc=21.0,
        metallicity=-1.3,
        age_yr=11e9,
        pm_phi1_range=(-3.5, -1.0),
        pm_phi2_range=(-1.5, 1.0),
        notes="Erkal+2017; Price-Whelan+2019; Bonaca+2020",
    )

    GJOLL = StreamConfig(
        name="Gjöll",
        origin_ra=96.510,
        origin_dec=-26.264,
        R=np.array(
            [
                [0.1699711720, -0.3630517498, -0.9161387598],
                [0.7476198980, 0.6467702272, -0.1517548985],
                [0.6477268397, -0.6688698877, 0.3644422090],
            ]
        ),
        phi1_range=(-60, 60),
        phi2_range=(-5, 5),
        distance_kpc=3.2,
        metallicity=-1.5,
        age_yr=12e9,
        pm_phi1_range=(2.0, 12.0),
        pm_phi2_range=(-4.0, 4.0),
        notes="Ibata+2019; Riley & Strigari 2020; Hansen+2020",
    )

    PHLEGETHON = StreamConfig(
        name="Phlegethon",
        origin_ra=321.887,
        origin_dec=-20.125,
        R=np.array(
            [
                [0.3007858380, -0.5254752584, -0.7953726741],
                [0.8966862028, 0.4396067893, 0.0485805598],
                [0.3241028159, -0.7284483144, 0.6042519802],
            ]
        ),
        phi1_range=(-40, 40),
        phi2_range=(-5, 5),
        distance_kpc=3.4,
        metallicity=-1.6,
        age_yr=12e9,
        pm_phi1_range=(5.0, 15.0),
        pm_phi2_range=(-3.0, 3.0),
        notes="Ibata+2018",
    )

    SYLGR = StreamConfig(
        name="Sylgr",
        origin_ra=175.497,
        origin_dec=-4.405,
        R=np.array(
            [
                [-0.0766375998, 0.0547233998, -0.9955694814],
                [-0.5818891546, -0.8132461547, 0.0126975590],
                [-0.8095984995, 0.5793064244, 0.0930827627],
            ]
        ),
        phi1_range=(-20, 20),
        phi2_range=(-5, 5),
        distance_kpc=3.6,
        metallicity=-2.9,
        age_yr=12e9,
        pm_phi1_range=(-8.0, 0.0),
        pm_phi2_range=(-4.0, 2.0),
        notes="Ibata+2019b; Roederer & Gnedin 2019",
    )
