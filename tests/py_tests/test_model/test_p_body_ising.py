from muca.cpp_muca import cpp_model

def test_cpp_model_p_body_two_dim_ising():
    model = cpp_model.PBodyTwoDimIsing(
        J=-1.4,
        p=3,
        Lx=4,
        Ly=4,
        spin=1.5,
        spin_scale_factor=1.0,
    )

    assert model.J == -1.4
    assert model.p == 3
    assert model.Lx == 4
    assert model.Ly == 4
    assert model.spin == 1.5
    assert model.spin_scale_factor == 1.0