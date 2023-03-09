"""Code to export gradient descent sequences into a movie."""
from typing import Optional

import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np


def write_movie(
    em_list: list,
    name: Optional[str] = "gauss_movie",
    xlim: Optional[int] = 4,
    ylim: Optional[int] = 4,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
):
    """Write the optimization steps into a mp4-movie file.

    Args:
        em_list (list): A list with the em-step outputs.
        name (str, optional): The name of the movie file. Defaults to "gauss_movie".
        xlim (int, optional): Largest x value in the data. Defaults to 4.
        ylim (int, optional): Largest y value in the data. Defaults to 4.
        xlabel (str, optional): The label for the x axis.
        ylabel (str, optional): The label for the y axis.

    Raises:
        RuntimeError: If ffmpeg is not loaded correctly.
    """
    try:
        ffmpeg_writer = manimation.writers["ffmpeg"]
    except RuntimeError:
        raise RuntimeError(
            "RuntimeError: If you are using anaconda or miniconda there might "
            "be a missing package named ffmpeg. Try installing it with "
            "'conda install -c conda-forge ffmpeg' in your terminal."
        )

    metadata = dict(
        title="Gaussian mixture fit",
        artist="Matplotlib",
        comment="Gauss bei der Arbeit.",
    )
    writer = ffmpeg_writer(fps=5, metadata=metadata)

    fig = plt.figure()

    with writer.saving(fig, f"{name}.gif", 100):
        for step in em_list:
            params_list, points, labels = step
            plt.plot(points[labels == 0][:, 0], points[labels == 0][:, 1], "g.")
            plt.plot(points[labels == 1][:, 0], points[labels == 1][:, 1], "c.")
            plt.plot(params_list[0][1][0], params_list[0][1][1], "gs")
            plt.plot(params_list[1][1][0], params_list[1][1][1], "cs")

            c1 = plt.Circle(
                xy=(params_list[0][1][0], params_list[0][1][1]),
                radius=np.linalg.norm(params_list[0][2]),
                color="g",
                fill=False,
            )
            fig.gca().add_patch(c1)
            c2 = plt.Circle(
                xy=(params_list[1][1][0], params_list[1][1][1]),
                radius=np.linalg.norm(params_list[1][2]),
                color="c",
                fill=False,
            )
            fig.gca().add_patch(c2)

            if xlabel:
                plt.xlabel(xlabel)
            if ylabel:
                plt.ylabel(ylabel)
            plt.xlim(-xlim, xlim)
            plt.ylim(-ylim, ylim)
            #            l.set_data(pos[0], pos[1])
            writer.grab_frame()
