from pathlib import Path

from water_mlip_benchmark.plots import plot_metric_relationship


def test_plot_metric_relationship(tmp_path: Path) -> None:
    output = tmp_path / "force_mae_vs_rdf_error.png"
    plot_metric_relationship(
        x=[0.3, 0.2, 0.1],
        y=[1.0, 0.6, 0.4],
        labels=["early", "mid", "best"],
        x_label="Force MAE",
        y_label="RDF error",
        title="Static error vs RDF error",
        output_path=output,
    )

    assert output.exists()
    assert output.stat().st_size > 0
