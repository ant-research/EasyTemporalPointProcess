def ode_update_op(z0, dz, dt):
    """
    General update operation for solving ODEs.

    Args:
        z0: Tensor or a list for Tensor whose shape is [..., dim]
            State at t0.
        dz: Tensor or a list for Tensor whose shape is [..., dim]
            Differentiation of state.
        dt: Tensor with shape [..., 1]
            Equal to t1 - t0.

    Returns:

    """
    if isinstance(z0, list) or isinstance(z0, tuple):
        return [item_z + dt * item_dz for item_z, item_dz in zip(z0, dz)]
    else:
        return z0 + dt * dz


def euler_step_method(diff_func, dt, z0):
    """
    Euler method for solving ODEs.

    Args:
        diff_func: function(state)
            Differential equation.
        dt: Tensor with shape [..., 1]
            Equal to t1 - t0.
        z0: Tensor or a list for Tensor whose shape is [..., dim]
            State at t0.

    Returns:
        Tensor or a list for Tensor whose shape is [..., dim], which is updated state.
    """
    dz = diff_func(z0)
    return ode_update_op(z0, dz, dt)


def rk2_step_method(diff_func, dt, z0):
    """
    Second order Runge-Kutta method for solving ODEs.

    Args:
        diff_func: function(dt, state)
            Differential equation.
        dt: Tensor with shape [..., 1]
            Equal to t1 - t0.
        z0: Tensor or a list for Tensor whose shape is [..., dim]
            State at t0.

    Returns:
        Tensor or a list for Tensor whose shape is [..., dim]
    """
    # shape -> [..., dim]
    k1 = diff_func(z0)
    k2 = diff_func(ode_update_op(z0, k1, dt))

    if isinstance(z0, list) or isinstance(z0, tuple):
        return [item_z + (item_k1 + item_k2) * dt * 0.5 for item_z, item_k1, item_k2 in zip(z0, k1, k2)]
    else:
        return z0 + dt * (k1 + k2) * 0.5


def rk4_step_method(diff_func, dt, z0):
    """
    Fourth order Runge-Kutta method for solving ODEs.

    Args:
        diff_func: function(dt, state)
            Differential equation.
        dt: Tensor with shape [..., 1]
            Equal to t1 - t0.
        z0: Tensor with shape [..., dim]
            State at t0.

    Returns:
        Tensor with shape [..., dim], which is updated state.
    """
    # shape -> [..., dim]
    k1 = diff_func(z0)
    k2 = diff_func(ode_update_op(z0, k1, dt / 2.0))
    k3 = diff_func(ode_update_op(z0, k2, dt / 2.0))
    k4 = diff_func(ode_update_op(z0, k3, dt))

    if isinstance(z0, list) or isinstance(z0, tuple):
        return [item_z + (item_k1 + 2.0 * item_k2 + 2.0 * item_k3 + item_k4) * dt / 6.0
                for item_z, item_k1, item_k2, item_k3, item_k4 in zip(z0, k1, k2, k3, k4)]
    else:
        return z0 + dt * (k1 + k2 * 2.0 + k3 * 2.0 + k4) / 6.0
