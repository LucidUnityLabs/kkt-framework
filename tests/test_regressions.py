from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import mpmath as mp
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

from kkt import core as c
from kkt.data import Galaxy, load_sparc
from kkt.fitting import (Minimum, Policy, bounded_minimum, feasible_ml_bounds,
                         fit_galaxy, minimum_2d, paired_delta, profile_intervals)
from kkt.statistics import (bootstrap_log_median, dex_errors, evolution_fit,
    forecast_sample_size, gls, log10_a_variance, log10_btfr_a)
from kkt.verification import Ledger, write_json

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("x", [0., 1e-300, 1e-20, 1e-10, .5, 1., 10., 1e100, 1e300])
def test_mu_oracle(x):
    with mp.workdps(90):
        a = mp.mpf(x)
        expected = 2*a/(mp.sqrt(1+4*a*a)+1)
        assert c.mu_kk(x) == pytest.approx(float(expected), rel=5e-15, abs=0)


@pytest.mark.parametrize("y", np.logspace(-300, 300, 13))
def test_inverse_mu_nu(y):
    x = math.sqrt(y)*math.hypot(math.sqrt(y), 1)
    assert c.mu_kk(x)*c.nu_kk(y) == pytest.approx(1., rel=2e-14)


@pytest.mark.parametrize("model,beta", [("kk",1), ("mond",1), ("gauge",1), ("generalized",.3), ("generalized",3)])
@pytest.mark.parametrize("g", [0., 1e-300, 1e-10, 1., 1e300])
def test_acceleration_oracle(model,beta,g):
    with mp.workdps(90):
        n, a, b = mp.mpf(g), mp.mpf(c.A0), mp.mpf(beta)
        expected = {"kk": lambda: mp.sqrt(n*n+n*a),
                    "mond": lambda: n/2+mp.sqrt(n*n/4+n*a),
                    "gauge": lambda: n+mp.sqrt(n*a),
                    "generalized": lambda: (n**(2*b)+(n*a)**b)**(1/(2*b))}[model]()
        assert c.acceleration(g, model=model, beta=beta) == pytest.approx(float(expected), rel=3e-13, abs=0)


@pytest.mark.parametrize("g", [1e-300, 1e-10, 1., 1e100, 1e300])
def test_delta_no_cancellation(g):
    with mp.workdps(90):
        n, a = mp.mpf(g), mp.mpf(c.A0)
        expected = a/(mp.sqrt(1+a/n)+1)
        assert c.delta_kk(g) == pytest.approx(float(expected), rel=3e-13, abs=0)


def test_beta_half_is_gauge_not_simple_mond():
    assert c.acceleration(c.A0, model="generalized", beta=.5)/c.A0 == pytest.approx(2)
    assert c.acceleration(c.A0, model="mond")/c.A0 == pytest.approx((1+math.sqrt(5))/2)


@pytest.mark.parametrize("x", [1e-12, 1e-9, 1e-5, .0099, .01, .0101, .1, 1., 100., 1e4])
def test_F_oracle(x):
    with mp.workdps(100):
        a = mp.mpf(x)
        expected = a*mp.sqrt(1+4*a*a)/2 + mp.asinh(2*a)/4 - a
        assert c.F_aqual(x) == pytest.approx(float(expected), rel=2e-11, abs=0)


@pytest.mark.parametrize("x", [1e-9, .01, .1, 1., 10.])
def test_F_scaled_quadrature_and_momentum(x):
    integral, error = c.F_aqual_integral(x)
    assert error < 1e-10*integral
    assert c.F_aqual(x) == pytest.approx(integral, rel=1e-10, abs=0)
    assert c.x_from_momentum(c.momentum(x)) == pytest.approx(x, rel=1e-13, abs=0)


@pytest.mark.parametrize("value", [-1., float("nan"), float("inf")])
def test_invalid_inputs_fail(value):
    for function in (c.mu_kk, c.nu_kk, c.acceleration, c.delta_kk):
        with pytest.raises(ValueError):
            function(value)


def test_nu_zero_not_arbitrarily_clipped():
    with pytest.raises(ValueError):
        c.nu_kk(0)
    assert c.acceleration(0) == 0
    assert c.mu_kk(0) == 0


def test_small_y_sign_and_remainder():
    y = 1e-4
    exact = c.nu_kk(y)*math.sqrt(y)
    assert exact > 1
    assert abs(exact-(1+y/2-y*y/8)) < y**3/10


def test_fiducial_cosmology_and_units():
    assert c.E_z(0) == 1
    assert c.E_z(0, omega_r=9e-5) == 1
    assert c.a0_z(2)/c.A0 == pytest.approx(math.sqrt(9.19))
    dex, percent = c.velocity_shift(2)
    assert percent == pytest.approx(100*(10**dex-1))
    assert percent == pytest.approx(31.951532619327928)
    with pytest.raises(ValueError):
        c.E_z(-1)
    with pytest.raises(ValueError):
        c.E_z(0, omega_m=.8, omega_r=.3)


def test_thermal_and_planck5():
    assert c.occupation() == pytest.approx(7.157165835186059e-18, rel=1e-14, abs=0)
    assert c.occupation(1000) == 0
    L = c.C/(2*c.H0)
    g5 = c.G*L
    length, mass = c.planck5(g5)
    assert mass**3 == pytest.approx(c.HBAR*c.HBAR/g5, rel=1e-13, abs=0)
    assert mass*c.C*length == pytest.approx(c.HBAR, rel=1e-14, abs=0)
    assert mass**3/(c.HBAR/(c.C*L)*(c.HBAR*c.C/c.G)) == pytest.approx(1.)


def test_perihelion_matches_independent_orbit_integration():
    e, A = .2, 1e-5  # dimensionless GM=a=1; small extra inward acceleration
    initial = [1-e, 0, 0, math.sqrt((1+e)/(1-e))]
    def rhs(t, state):
        x, y, vx, vy = state
        radius = math.hypot(x,y)
        factor = -(radius**-3 + A/radius)
        return [vx,vy,factor*x,factor*y]
    def periapsis(t, state):
        return state[0]*state[2]+state[1]*state[3]
    periapsis.direction = 1
    result = solve_ivp(rhs, (0, 7), initial, method="DOP853", rtol=2e-11,
                       atol=2e-13, events=periapsis, max_step=.05)
    assert result.success
    state = result.y_events[0][-1]
    angle = math.atan2(state[1],state[0])
    predicted = c.perihelion_per_orbit(1,e,-A,gm=1)
    assert angle < 0
    assert angle == pytest.approx(predicted, rel=3e-4)


def test_ledger_tiny_numbers_shapes_nonfinite_and_exit():
    ledger=Ledger()
    ledger.close("wrong-tiny",1e-11,1.042e-10,rtol=1e-10)
    ledger.close("nan",np.nan,1.,rtol=1e-10)
    ledger.close("shape",[1.],1.,rtol=1e-10)
    ledger.close("good-zero",0.,0.,rtol=0)
    ledger.truth("boolean",True)
    ledger.not_verified("physics","missing metric")
    assert ledger.report()["counts"] == {"PASS":2,"FAIL":3,"NOT_VERIFIED":1}
    assert ledger.exit_code() == 1
    ledger2=Ledger(); ledger2.not_verified("p","missing data")
    assert ledger2.exit_code() == 0
    assert ledger2.exit_code(require_all_claims=True) == 2
    with pytest.raises(ValueError):
        ledger2.close("p",1,1,rtol=1e-10)
    with pytest.raises(ValueError):
        ledger.close("bad-tolerance",1,1,rtol=float("nan"))


def test_optimizer_scale_and_endpoints():
    objective=lambda a: ((a-1.3e-10)/1e-10)**2
    original=minimize_scalar(objective,bounds=(.6e-10,2e-10),method="bounded")
    assert original.nfev == 1
    assert original.fun > .01
    fixed=bounded_minimum(lambda alpha: objective(alpha*1e-10),(.6,2.))
    assert fixed.x == pytest.approx(1.3,abs=1e-8)
    assert not fixed.boundary
    boundary=bounded_minimum(lambda x:x*x,(.6,2.))
    assert boundary.x == .6 and boundary.boundary
    with pytest.raises(FloatingPointError):
        bounded_minimum(lambda x:float("nan"),(0,1))


def test_profile_crossings_and_truncation():
    f=lambda x:((x-.65)/.1)**2
    best=bounded_minimum(f,(.1,1.2))
    interval=profile_intervals(f,best,(.1,1.2))["intervals"][0]
    radius=.1*math.sqrt(chi2.ppf(.95,1))
    assert interval["lower"] == pytest.approx(.65-radius,abs=1e-9)
    assert interval["upper"] == pytest.approx(.65+radius,abs=1e-9)
    assert not interval["lower_truncated"]
    truncated=profile_intervals(lambda x:(x/.1)**2,Minimum(0,0,True,False),(0,1))["intervals"][0]
    assert truncated["lower_truncated"]


def test_disconnected_support_and_2d_minimum():
    f=lambda x:(x*x-1)**2/.01
    support=profile_intervals(f,Minimum(-1,0,False,False),(-2,2))["intervals"]
    assert len(support)==2
    best=minimum_2d(lambda b,a:(b-1.1)**2+(a-.7)**2,(.3,2),(.2,2),grid_points=5)
    assert best["beta"]==pytest.approx(1.1,abs=1e-6)
    assert best["alpha"]==pytest.approx(.7,abs=1e-6)
    assert not best["boundary"]
    # A nearby scale inside a contour is NOT the fixed point (1,1).
    q=lambda a:((a-.9)/.02)**2
    assert q(.9)<chi2.ppf(.95,2) and q(1)>chi2.ppf(.95,2)


def example_galaxy():
    radius=np.array([1.,2.,3.])*c.KPC
    gas=np.array([-.2,.1,.1])*c.A0
    disk=np.array([1.,.6,.3])*c.A0
    observed=np.sqrt(c.acceleration(gas+.7*disk)*radius)
    return Galaxy("fixture",("f:1","f:2","f:3"),radius,observed,
                  np.full(3,1000.),gas,disk,np.zeros(3),np.array([30.,20.,10.]))


def test_fixed_observations_physical_bounds_fit_and_alignment():
    gal=example_galaxy(); policy=Policy()
    lo,hi=feasible_ml_bounds(gal,policy)
    assert lo>=.2 and hi==10
    fitted=fit_galaxy(gal,policy=policy)
    assert fitted["ml"] == pytest.approx(.7,abs=1e-6)
    assert fitted["n"]==3 and fitted["row_ids"]==list(gal.row_ids)
    assert len(fitted["residuals"])==3
    assert np.max(np.abs(paired_delta(fitted,fitted)))==0
    altered=dict(fitted,row_ids=list(reversed(fitted["row_ids"])))
    with pytest.raises(ValueError):
        paired_delta(fitted,altered)
    with pytest.raises(ValueError):
        gal.radius_m[0]=1


TSV = """# sample rows transcribed from the audited TSV, not a full dataset
Name\tDist\tRad\tVobs\te_Vobs\tVgas\tVdisk\tSBdisk
 \tMpc\tkpc\tkm/s\tkm/s\tkm/s\tkm/s\tLsun/pc2
-----------\t------\t------\t------\t-----\t------\t------\t-------
DDO064\t6.80\t0.10\t6.29\t4.62\t-1.13\t1.96\t28.50
DDO064\t6.80\t0.30\t13.90\t4.62\t-2.45\t6.31\t27.26
DDO064\t6.80\t0.49\t15.60\t4.62\t-0.64\t9.67\t24.99
"""


def test_parser_requires_bulge_and_preserves_signed_gas(tmp_path):
    path=tmp_path/"sample.tsv"; path.write_text(TSV)
    with pytest.raises(ValueError,match="Vbulge"):
        load_sparc(path)
    galaxies,meta=load_sparc(path,allow_disk_only=True)
    assert len(galaxies)==1 and meta["points"]==3
    assert galaxies[0].gas_acceleration[0]<0
    assert galaxies[0].bulge_acceleration[0]==0
    assert "diagnostic" in meta["baryonic_model"]
    assert galaxies[0].sb_disk[0]==28.5  # column is NOT Vbulge


@pytest.mark.parametrize("old,new",[("6.29","nan"),("0.10","0.00"),("4.62","0.00"),
                                    ("km/s","m/s"),("6.29","garbage")])
def test_malformed_data_never_silently_skipped(tmp_path,old,new):
    path=tmp_path/"sample.tsv"; path.write_text(TSV.replace(old,new,1))
    with pytest.raises(ValueError):
        load_sparc(path,allow_disk_only=True)


def test_empty_and_duplicate_data_fail(tmp_path):
    path=tmp_path/"sample.tsv"; path.write_text("# empty\n")
    with pytest.raises(ValueError): load_sparc(path,allow_disk_only=True)
    path.write_text(TSV+TSV.splitlines()[-1]+"\n")
    with pytest.raises(ValueError,match="duplicate"): load_sparc(path,allow_disk_only=True)


def test_log_btfr_and_dex_conversion():
    v,m=100.,10.
    expected=(v*1e3)**4/(c.G*10**m*c.M_SUN)
    assert 10**log10_btfr_a(v,m)==pytest.approx(expected)
    lower,upper=dex_errors(1.,.15)
    assert lower==pytest.approx(1-10**(-.15))
    assert upper==pytest.approx(10**.15-1)
    assert upper>.4 and lower>.29
    assert log10_a_variance(.1,.2,0)==pytest.approx(.20)
    with pytest.raises(ValueError): log10_a_variance(.1,.2,.03)


def test_constant_data_does_not_detect_evolution():
    z=np.array([0.,.5,1.,2.])
    result=evolution_fit(z,np.full(4,-10.),np.eye(4)*.05**2)
    assert abs(result["gamma"])<1e-12
    assert result["nested_delta_chi2"]<1e-20
    assert result["nominal_nested_p_value"]>.999999
    assert result["free_dof"]==2
    assert result["fixed_dof"]==4
    assert result["free_normalization_dof"]==3


def test_injected_evolution_and_H0_linear_scale():
    z=np.array([0.,.5,1.,2.,3.])
    y=np.log10(4*c.a0_z(z))
    result=evolution_fit(z,y,np.eye(5)*.01**2)
    assert result["gamma"]==pytest.approx(1.,abs=1e-12)
    assert result["kk_alpha"]==pytest.approx(4.)
    assert result["conditional_H0_km_s_Mpc"]==pytest.approx(4*67.4)


def test_gls_covariance_and_degenerate_design():
    y=np.array([1.,2.,3.]); X=np.column_stack([np.ones(3),[0,1,2]])
    with pytest.raises(ValueError): gls(X,y,np.zeros((3,3)))
    with pytest.raises(ValueError): gls(np.ones((3,2)),y,np.eye(3))
    with pytest.raises(ValueError): evolution_fit([1,1,1],[-10,-10,-10],np.eye(3))
    with pytest.raises(ValueError): gls(X,y,np.array([[1,.1,0],[0,1,0],[0,0,1]]))


def test_bootstrap_and_power_forecast():
    a=bootstrap_log_median([-10,-9.9,-10.1,-10.05],repetitions=200)
    b=bootstrap_log_median([-10,-9.9,-10.1,-10.05],repetitions=200)
    assert a==b
    dlogv,_=c.velocity_shift(2)
    n=forecast_sample_size(dlogv,sigma_independent=.11)
    assert n==21
    assert forecast_sample_size(dlogv,sigma_independent=.11,power=.9)>n
    assert forecast_sample_size(dlogv,sigma_independent=.11,sigma_systematic=.03) is None
    assert forecast_sample_size(0,sigma_independent=.11) is None


def test_json_rejects_nan_and_is_atomic(tmp_path):
    path=tmp_path/"result.json"; write_json(path,{"ok":1})
    with pytest.raises(ValueError): write_json(path,{"bad":float("nan")})
    assert json.loads(path.read_text())=={"ok":1}
    assert len(list(tmp_path.iterdir()))==1


@pytest.mark.parametrize("filename", ["kk_verify_all.py","kk_solar_system.py","kk_z_dependence.py",
    "kk_dimensional_reduction.py","sparc_tests_abc.py","kk_dS_force_law.py",
    "kk_rar_morphology.py","kk_z_btfr_data.py"])
def test_legacy_import_has_no_analysis_side_effects(filename,capsys):
    spec=importlib.util.spec_from_file_location("legacy_"+filename[:-3],ROOT/"scripts"/filename)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    captured=capsys.readouterr()
    assert captured.out==captured.err==""


def test_cli_from_unrelated_cwd_and_strict_gate(tmp_path):
    script=ROOT/"scripts"/"kk_verify_all.py"
    result=subprocess.run([sys.executable,str(script),"--out",str(tmp_path/"normal")],
                           cwd=tmp_path,text=True,capture_output=True)
    assert result.returncode==0, result.stderr
    output=json.loads((tmp_path/"normal"/"verify.json").read_text())
    assert output["counts"]=={"FAIL":0,"NOT_VERIFIED":8,"PASS":73}
    result=subprocess.run([sys.executable,str(script),"--require-all-claims","--out",str(tmp_path/"strict")],
                           cwd=tmp_path,text=True,capture_output=True)
    assert result.returncode==2


def test_missing_btfr_inputs_are_blocked_not_fabricated(tmp_path):
    result=subprocess.run([sys.executable,str(ROOT/"scripts"/"kk_z_btfr_data.py"),
                           "--btfr-data",str(tmp_path/"missing.json"),"--out",str(tmp_path/"out")],
                           cwd=tmp_path,text=True,capture_output=True)
    assert result.returncode==2
    assert json.loads((tmp_path/"out"/"btfr.json").read_text())["status"]=="BLOCKED"

def test_empty_verification_arrays_fail():
    ledger=Ledger(); ledger.close("empty",[],[],rtol=1e-12)
    assert ledger.report()["counts"]["FAIL"]==1
    assert ledger.exit_code()==1


def test_complete_catalogue_matches_preserved_projection():
    complete, full_meta = load_sparc(ROOT/"data"/"rotation_curves_full.tsv")
    legacy, legacy_meta = load_sparc(ROOT/"data"/"rotation_curves.tsv", allow_disk_only=True)
    assert full_meta["galaxies"] == legacy_meta["galaxies"] == 175
    assert full_meta["points"] == legacy_meta["points"] == 3391
    assert "gas+disk+bulge" in full_meta["baryonic_model"]
    assert any(np.any(g.bulge_acceleration > 0) for g in complete)
    for full, old in zip(complete, legacy, strict=True):
        assert full.name == old.name
        for field in ("radius_m", "velocity_m_s", "velocity_error_m_s",
                      "gas_acceleration", "disk_acceleration", "sb_disk"):
            np.testing.assert_array_equal(getattr(full, field), getattr(old, field))


@pytest.mark.parametrize("target", [1e-4, 1-1e-4])
def test_optimizer_refines_interior_minimum_inside_boundary_grid_cell(target):
    result = bounded_minimum(lambda x: (x-target)**2, (0, 1))
    assert result.x == pytest.approx(target, abs=1e-8)
    assert result.cost < 1e-16
    assert not result.boundary


@pytest.mark.parametrize("scale", [1e-14, 1e-20])
def test_small_nonconstant_objective_refines_despite_flat_diagnostic(scale):
    target = .123456
    result = bounded_minimum(lambda x: scale*(x-target)**2, (0, 1))
    assert result.flat_on_grid
    assert abs(result.x-target) < 1e-7
    assert result.cost < scale*1e-14


def test_constant_objective_has_no_identifiable_sampled_basin(monkeypatch):
    import kkt.fitting as fitting
    def unexpected(*args, **kwargs):
        raise AssertionError("constant sampled objective must not invent a basin")
    monkeypatch.setattr(fitting, "minimize_scalar", unexpected)
    result = bounded_minimum(lambda x: 7., (0, 1))
    assert result.flat_on_grid
    assert result.cost == 7.
    assert result.x == 0.
    assert result.boundary


def test_declared_search_controls_reach_catalogue_search_and_report(tmp_path, monkeypatch):
    import kkt.commands as commands
    policies = []
    captured = {}
    monkeypatch.setattr(commands, "load_sparc", lambda *a, **kw: ([], {"points": 0, "galaxies": 0}))
    def sample(galaxies, *, policy, model, beta, a0):
        policies.append(policy.ml_grid_points)
        return {"cost": (beta-.8)**2+(a0/commands.c.A0-1.2)**2}
    monkeypatch.setattr(commands, "fit_sample", sample)
    monkeypatch.setattr(commands, "provenance", lambda: {})
    monkeypatch.setattr(commands, "write_json", lambda path, value: captured.update(value))
    assert commands.main(["ds", "--out", str(tmp_path), "--ml-grid-points", "19",
                          "--scalar-grid-points", "7", "--profile-grid-points", "9",
                          "--joint-grid-points", "3", "--joint-starts", "2"]) == 0
    assert set(policies) == {19}
    controls = captured["search_controls"]
    assert controls["scalar_grid_points"] == 7
    assert controls["profile_grid_points"] == 9
    assert captured["searched_minimum"]["grid_points_per_axis"] == 3
    assert captured["searched_minimum"]["refined_starts"] == 2
    assert abs(captured["searched_minimum"]["beta"]-.8) < 1e-6


@pytest.mark.parametrize("flag,value", [("--ml-grid-points", "4"),
                                        ("--joint-starts", "0"),
                                        ("--joint-grid-points", "2")])
def test_cli_rejects_invalid_search_resolution(flag, value):
    from kkt.commands import main
    with pytest.raises(SystemExit) as exc:
        main(["verify", flag, value])
    assert exc.value.code == 2


@pytest.mark.parametrize("command", ["sparc", "ds"])
def test_custom_domains_and_executed_search_provenance(command, tmp_path, monkeypatch):
    import kkt.commands as commands
    galaxy = example_galaxy()
    monkeypatch.setattr(commands, "load_sparc", lambda *a, **kw:
                        ([galaxy], {"points": 3, "galaxies": 1}))
    def sample(galaxies, *, policy, model="kk", beta=1., a0=c.A0):
        return {"cost": (beta-.8)**2+(a0/c.A0-1.2)**2, "nominal_dof": 2,
                "fits": [{"name": galaxy.name, "row_ids": list(galaxy.row_ids),
                          "residuals": [0., 0., 0.]}]}
    monkeypatch.setattr(commands, "fit_sample", sample)
    monkeypatch.setattr(commands, "provenance", lambda: {})
    assert commands.main([command, "--out", str(tmp_path),
        "--alpha-bounds", ".61", "1.73", "--beta-bounds", ".42", "1.89",
        "--objective", "acceleration", "--floor", ".07", "--ml-min", ".31",
        "--ml-max", "4.2", "--bulge-ml-ratio", "1.6", "--ml-grid-points", "5",
        "--scalar-grid-points", "5", "--profile-grid-points", "7",
        "--joint-grid-points", "3", "--joint-starts", "1"]) == 0
    result = json.loads((tmp_path/f"{command}.json").read_text())
    controls = result["search_controls"]
    requested, executed = controls["requested"], controls["executed"]
    assert requested["alpha_bounds"] == [.61, 1.73]
    assert requested["beta_bounds"] == [.42, 1.89]
    assert requested["policy"] == result["input"]["policy"] == {
        "objective": "acceleration", "fractional_floor": .07, "ml_min": .31,
        "ml_max": 4.2, "bulge_ml_ratio": 1.6, "ml_grid_points": 5}
    assert executed["galaxy_ml"]["policy"] == requested["policy"]
    assert executed["galaxy_ml"]["feasible_domains"] == [{"name": "fixture", "bounds": [.31, 4.2]}]
    assert executed["galaxy_ml"]["xatol"] == 1e-8
    assert executed["galaxy_ml"]["maxiter"] == 1000
    assert controls["execution_status"] == "completed"
    if command == "sparc":
        assert set(executed) == {"galaxy_ml", "mond_free_alpha", "generalized_free_beta", "beta_profile_support"}
        assert executed["mond_free_alpha"]["alpha_bounds"] == [.61, 1.73]
        assert executed["generalized_free_beta"]["beta_bounds"] == [.42, 1.89]
        assert executed["beta_profile_support"]["beta_bounds"] == [.42, 1.89]
        assert executed["beta_profile_support"]["root_xtol"] == 1e-10
        assert executed["beta_profile_support"]["root_rtol"] == 1e-12
    else:
        assert set(executed) == {"galaxy_ml", "generalized_joint", "beta1_free_alpha", "fixed_KK_evaluation"}
        assert executed["generalized_joint"]["alpha_bounds"] == [.61, 1.73]
        assert executed["generalized_joint"]["beta_bounds"] == [.42, 1.89]
        assert executed["generalized_joint"]["xtol"] == 1e-7
        assert executed["generalized_joint"]["ftol"] == 1e-10
        assert executed["beta1_free_alpha"]["alpha_bounds"] == [.61, 1.73]
        assert requested["profile_grid_points"] == 7
        assert "beta_profile_support" not in executed


def test_blocked_search_does_not_claim_executed_controls(tmp_path, monkeypatch):
    import kkt.commands as commands
    def blocked(*a, **kw):
        raise ValueError("tiny missing input")
    monkeypatch.setattr(commands, "load_sparc", blocked)
    monkeypatch.setattr(commands, "provenance", lambda: {})
    assert commands.main(["ds", "--out", str(tmp_path)]) == 2
    controls = json.loads((tmp_path/"ds.json").read_text())["search_controls"]
    assert controls["executed"] == {}
    assert controls["execution_status"] == "not recorded; command blocked"
