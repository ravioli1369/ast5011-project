import os
import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp
from astropy import units as u
from astropy.coordinates import SkyCoord
from ezpadova import parsec
from gala.coordinates import GreatCircleICRSFrame
from galstreams import MWStreams

params = {
    "text.usetex": True,
    "font.family": "serif",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.top": True,
    "ytick.left": True,
    "ytick.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.size": 2.5,
    "xtick.major.size": 5,
    "ytick.minor.size": 2.5,
    "ytick.major.size": 5,
    "axes.axisbelow": False,
    "figure.dpi": 300,
}
plt.rcParams.update(params)

warnings.filterwarnings(
    "ignore", category=pd.errors.SettingWithCopyWarning, append=True
)

ISOCHRONES_DIR = "./isochrones"
MWS = MWStreams()


@dataclass
class StreamConfig:
    """
    Configuration for extracting a stellar stream from Gaia.
    """

    name: str
    metallicity: float | None = None  # [Fe/H]
    age_yr: float | None = None  # years
    pm_phi1_range: tuple[float, float] | None = None  # mas/yr
    pm_phi2_range: tuple[float, float] | None = None  # mas/yr
    g_mag_range: tuple[float, float] = (14, 21)
    ruwe_max: float = 1.4
    parallax_max: float = 1.0  # mas
    notes: str = ""

    def __post_init__(self):
        self._galstreams_properties()
        if self.metallicity is not None and self.age_yr is not None:
            self.isochrone = self._get_isochrone()
            self.cmd_polygon, self.cmd_polygon_full = self._build_cmd_polygon()

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
        self.track = track.track
        self.track_phi = self.track.transform_to(self.stream_frame)
        self.phi1_range: tuple[float, float] = (
            self.track_phi.phi1.min().deg - 5,
            self.track_phi.phi1.max().deg + 5,
        )
        self.phi2_range: tuple[float, float] = (
            self.track_phi.phi2.min().deg - 2.5,
            self.track_phi.phi2.max().deg + 2.5,
        )
        self.ra_range: tuple[float, float] = (
            self.track.ra.min().deg - 5,
            self.track.ra.max().deg + 5,
        )
        self.dec_range: tuple[float, float] = (
            self.track.dec.min().deg - 2.5,
            self.track.dec.max().deg + 2.5,
        )
        self.distance_kpc: float = float(np.median(self.track.distance.kpc))

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
        poly_full = np.column_stack(
            [
                np.concatenate(
                    [
                        self.isochrone["BP_RP"] - color_buffer,
                        (self.isochrone["BP_RP"] + color_buffer)[::-1],
                        [self.isochrone["BP_RP"].iloc[0] - color_buffer],
                    ]
                ),
                np.concatenate(
                    [
                        self.isochrone["Gmag"],
                        self.isochrone["Gmag"][::-1],
                        [self.isochrone["Gmag"].iloc[0]],
                    ]
                ),
            ]
        )
        return poly, poly_full


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

    PHLEGETHON = StreamConfig(
        name="Phlegethon",
        metallicity=-1.4,
        age_yr=10e9,
        notes="Ibata+2018",
    )

    JHELUM = StreamConfig(
        name="Jhelum",
        notes="Ibata+2023",
    )

    INDUS = StreamConfig(
        name="Indus",
        notes="Ibata+2023",
    )

    NGC6397 = StreamConfig(
        name="NGC6397",
        notes="Ibata+2023",
    )

    C12 = StreamConfig(
        name="C-12",
        notes="Ibata+2023",
    )

    ORPHAN = StreamConfig(
        name="Orphan",
        notes="Ibata+2023",
    )

    YLGR = StreamConfig(
        name="Ylgr",
        notes="Ibata+2023",
    )


streamfinder_stars = pd.read_csv(
    "./streamfinder/streamfinder_stars.csv", comment="#", sep=r"\s+"
)
streamfinder_streamid = pd.read_csv(
    "./streamfinder/streamfinder_streamid.csv", comment="#", sep="\t"
)


def obtain_stream_data(stream: Streams) -> pd.DataFrame:
    stream_id = int(
        streamfinder_streamid[streamfinder_streamid["Name"] == stream.name][
            "s_ID"
        ].iloc[0]
    )
    stream_data = streamfinder_stars[streamfinder_stars["Stream"] == stream_id]
    stream_coords = SkyCoord(
        ra=stream_data["RAdeg"].values * u.degree,
        dec=stream_data["DEdeg"].values * u.degree,
        pm_ra_cosdec=stream_data["pmRA"].values * u.mas / u.yr,
        pm_dec=stream_data["pmDE"].values * u.mas / u.yr,
    ).transform_to(stream.stream_frame)
    stream_data["phi1"] = stream_coords.phi1.deg
    stream_data["phi2"] = stream_coords.phi2.deg
    stream_data["pm_phi1"] = stream_coords.pm_phi1_cosphi2.value
    stream_data["pm_phi2"] = stream_coords.pm_phi2.value
    return stream_data


def obtain_stream_density(
    stream_data: pd.DataFrame,
    PHI_XLIM: tuple,
    binstep: float = 1.5,
) -> pd.DataFrame:
    stream_data = stream_data[
        (stream_data["phi1"] >= PHI_XLIM[0]) & (stream_data["phi1"] <= PHI_XLIM[1])
    ]
    density_bins = np.arange(*PHI_XLIM, step=binstep)
    density_centers = 0.5 * (density_bins[:-1] + density_bins[1:])
    counts = np.histogram(stream_data["phi1"], bins=density_bins)[0].astype(float)
    median = np.median(counts)
    counts -= median
    counts_smooth = sp.ndimage.gaussian_filter1d(counts.astype(float), sigma=1.5)
    return pd.DataFrame({"phi1": density_centers, "density": counts_smooth})


def plot_density(
    stream: Streams,
    density_data: pd.DataFrame,
    stream_data: pd.DataFrame,
    PHI_XLIM: tuple,
    PHI_YLIM: tuple,
) -> None:
    fig, (ax_sky, ax_dens) = plt.subplots(
        2,
        1,
        figsize=(12, 5),
        height_ratios=[2, 1],
        sharex=True,
        gridspec_kw={"hspace": 0.05},
    )

    ax_sky.scatter(
        stream_data["phi1"],
        stream_data["phi2"],
        s=2,
        c="k",
        alpha=0.3,
    )
    ax_sky.plot(stream.track_phi.phi1, stream.track_phi.phi2, "r-", lw=1)
    ax_sky.set_ylim(*PHI_YLIM)
    ax_sky.set_ylabel(r"$\phi_2$ [deg]", fontsize=14)

    ax_dens.step(
        density_data["phi1"],
        density_data["density"],
        where="mid",
        color="steelblue",
        lw=1.5,
    )
    ax_dens.fill_between(
        density_data["phi1"],
        density_data["density"],
        step="mid",
        color="steelblue",
        alpha=0.3,
    )
    ax_dens.set_xlabel(r"$\phi_1$ [deg]", fontsize=14)
    ax_dens.set_ylabel(r"$\rho$ [counts]", fontsize=14)
    ax_dens.set_xlim(*PHI_XLIM)
    fig.savefig(f"./plots/{stream.name}_density_streamfinder.pdf", bbox_inches="tight")
    fig.show()
