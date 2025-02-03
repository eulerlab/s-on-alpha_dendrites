import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree


def pd_read_swc(swc_file):
    return pd.read_csv(swc_file, sep=r'\s+|\t+',
                       names=["n", "type", "x", "y", "z", "r", "parent"], header=None, comment='#', engine='python')


def pd_save_swc(df, swc_file, comment=None):
    df.to_csv(swc_file, sep=' ', header=None, index=False)
    # Add comment to beginning of file
    if comment is None:
        return
    with open(swc_file, 'r') as f:
        lines = f.readlines()
    with open(swc_file, 'w') as f:
        f.write(f'# {comment}\n')
        f.writelines(lines)


def plot_stack_and_vessels(df_lower, stack, soma_xyz_px, df_upper=None):
    proj_xy = np.max(stack, axis=2)
    proj_xz = np.max(stack, axis=1)
    proj_yz = np.max(stack, axis=0)

    plt.close('all')
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    ax = axs[0]
    ax.imshow(proj_xy.T, cmap='gray', origin='lower')
    ax.scatter(df_lower.x, df_lower.y, alpha=0.5, c='C0')
    if df_upper is not None:
        ax.scatter(df_upper.x, df_upper.y, alpha=0.5, c='C1')
    ax.scatter(soma_xyz_px[0], soma_xyz_px[1], marker='X', s=20, color='black')

    ax = axs[1]
    ax.imshow(proj_xz.T, cmap='gray', origin='lower')
    ax.scatter(df_lower.x, df_lower.z, alpha=0.5, c='C0')
    if df_upper is not None:
        ax.scatter(df_upper.x, df_upper.z, alpha=0.5, c='C1')
    ax.scatter(soma_xyz_px[0], soma_xyz_px[2], marker='X', s=20, color='black')

    ax = axs[2]
    ax.imshow(proj_yz.T, cmap='gray', origin='lower')
    ax.scatter(df_lower.y, df_lower.z, alpha=0.5, c='C0', label='Lower')
    if df_upper is not None:
        ax.scatter(df_upper.y, df_upper.z, alpha=0.5, c='C1', label='Upper')
    ax.scatter(soma_xyz_px[1], soma_xyz_px[2], marker='X', s=20, color='black')
    ax.legend(bbox_to_anchor=(0.5, 1.3), loc='lower center')

    plt.tight_layout()
    plt.show()


def get_n_splines(df, f_space=50, n_min=4):
    x_rng = df.x.max() - df.x.min()
    y_rng = df.y.max() - df.y.min()

    x_splines = np.maximum(n_min, int(np.round(x_rng / f_space)))
    y_splines = np.maximum(n_min, int(np.round(y_rng / f_space)))

    return x_splines, y_splines


def fit_gam(df, f_space=50, lam=0.001, plane='none', penalties='l2', verbose=True):
    from pygam import LinearGAM, te, l
    X = df[['x', 'y']].values
    z = df['z'].values

    x_splines, y_splines = get_n_splines(df, f_space=f_space)
    if verbose:
        print(f'Fitting GAM with {x_splines} x {y_splines} splines')

    if plane == 'none':
        gam = LinearGAM(te(0, 1, lam=lam, n_splines=(x_splines, y_splines)), penalties=penalties).fit(X, z)
    elif plane == 'linear':
        gam = LinearGAM(te(0, 1, lam=0, n_splines=[2, 2], spline_order=[1, 1])
                        + te(0, 1, lam=lam, n_splines=(x_splines, y_splines)), penalties=penalties).fit(X, z)
    elif plane == 'linear_only':
        gam = LinearGAM(te(0, 1, lam=0, n_splines=[2, 2], spline_order=[1, 1]), penalties=penalties).fit(X, z)
    else:
        raise ValueError(f'Unknown plane: {plane}')
    return gam


def plot_gam(df, gam, ax, scatter=True, color='C0'):
    X = df[['x', 'y']].values

    if scatter:
        ax.scatter(df.x, df.y, df.z, label='Data', c='k')
        ax.scatter(df.x, df.y, gam.predict(X), label='Pred', c=color)

    xgrid = np.linspace(np.min(df.x), np.max(df.x), 101)
    ygrid = np.linspace(np.min(df.y), np.max(df.y), 101)
    xx, yy = np.meshgrid(xgrid, ygrid)

    zpred = gam.predict(X=np.stack([xx.flat, yy.flat]).T)
    zz = np.reshape(zpred, xx.shape)
    ax.plot_surface(xx, yy, zz, alpha=0.5, color=color)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def plot_paths_3d(paths, ax, color='black'):
    for path in paths:
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color=color,
                solid_capstyle='round', solid_joinstyle='round')


def plot_fits(gam_lower, df_lower_bvs, gam_upper=None, df_upper_bvs=None, paths=None):
    # Plot the actual vs predicted values
    fig = plt.figure(figsize=(12, 4))

    ax = fig.add_subplot(131, projection='3d')
    ax.set_title('Lower')
    plot_gam(df_lower_bvs, gam_lower, ax, color='C0')

    ax = fig.add_subplot(132, projection='3d')
    if gam_upper is not None:
        ax.set_title('Upper')
        plot_gam(df_upper_bvs, gam_upper, ax, color='C1')
    else:
        ax.axis('off')

    ax = fig.add_subplot(133, projection='3d')
    plot_gam(df_lower_bvs, gam_lower, ax, scatter=False, color='C0')
    if df_upper_bvs is not None:
        plot_gam(df_upper_bvs, gam_upper, ax, scatter=False, color='C1')
    if paths is not None:
        plot_paths_3d(paths, ax)

    plt.tight_layout()
    plt.show()


def plot_paths_3d_flat(df_paths, df_paths_flat, soma_xyz_um):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    plot_paths_3d(df_paths.path, ax, color='k')
    plot_paths_3d(df_paths_flat.path, ax, color='r')
    ax.view_init(elev=0, azim=45)
    ax.scatter(*soma_xyz_um)
    plt.show()


def create_gam_grid(gam, x_min, x_max, y_min, y_max, dxy=2):
    """
    Create a grid for the GAM surface.
    """
    x = np.arange(x_min, x_max, dxy)
    y = np.arange(y_min, y_max, dxy)
    X, Y = np.meshgrid(x, y)

    xy_grid = np.column_stack([X.ravel(), Y.ravel()])

    Z = gam.predict(xy_grid)
    xyz_grid = np.column_stack([xy_grid, Z])
    Z = Z.reshape(X.shape)

    return X, Y, Z, xyz_grid


def shortest_distances_to_gam_grid(xyz_grid, points):
    """
    Calculate the shortest distances between points and xyz-grid.
    Create a KD-tree for efficient nearest neighbor search.
    Find the nearest grid points in x-y plane.
    Get the corresponding points on the GAM surface.
    """
    tree = cKDTree(xyz_grid)
    distances, indices = tree.query(points)
    closest_points = xyz_grid[indices]
    return closest_points, distances


def unwarp_swc(df_swc, gam_lower, df_lower_bvs, gam_upper, df_upper_bvs, pixel_size_um, z_step_um, dxy=2, plot=False):
    voxel_size_um = np.array([pixel_size_um, pixel_size_um, z_step_um])
    points_xyz = df_swc[['x', 'y', 'z']].values / voxel_size_um

    X, Y, Z, xyz_grid = create_gam_grid(
        gam_lower,
        points_xyz[:, 0].min() - 50, points_xyz[:, 0].max() + 50,
        points_xyz[:, 1].min() - 50, points_xyz[:, 1].max() + 50,
        dxy
    )

    closest_points, distances_px = shortest_distances_to_gam_grid(xyz_grid, points_xyz)
    distances_um = distances_px * z_step_um

    if plot:
        plot_gam_grid_distances(X, Y, Z, points_xyz, closest_points)

    xy_shared = create_grid_in_convex_hulls(
        points1=df_lower_bvs[['x', 'y']].values, points2=df_upper_bvs[['x', 'y']].values)

    # Using xy_shared, compute the distance between the upper and lower bands
    z_upper = gam_upper.predict(X=xy_shared)
    points_xyz_upper = np.column_stack([xy_shared, z_upper])
    _, distances_px_upper = shortest_distances_to_gam_grid(xyz_grid, points_xyz_upper)
    d_med_um = np.median(distances_px_upper) * z_step_um

    df_swc_flat = df_swc.copy()
    df_swc_flat["z"] = distances_um / d_med_um

    return df_swc_flat, d_med_um


def plot_gam_grid_distances(X, Y, Z, points, closest_points, max_points=100):
    """
    Plot the GAM grid surface and the distances to the closest points.
    """

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_box_aspect((1, 1, 1))
    ax.set_aspect('equal')

    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

    n_points = min(len(points), max_points)
    idxs = np.random.choice(np.arange(len(points)), n_points)

    for i in idxs:
        ax.scatter(*points[i], color='red', s=20)
        ax.scatter(*closest_points[i], color='green', s=20)
        ax.plot([points[i, 0], closest_points[i, 0]],
                [points[i, 1], closest_points[i, 1]],
                [points[i, 2], closest_points[i, 2]],
                color='black', linestyle='-', linewidth=0.5, ms=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'GAM Grid Surface with Vertical Distances to {n_points} Points')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    ax.set_box_aspect((1, 1, 1))
    ax.set_aspect('equal')

    plt.show()


def create_grid_in_convex_hulls(points1, points2, n=100):
    # Create convex hulls for both sets of points
    hull1 = ConvexHull(points1)
    hull2 = ConvexHull(points2)

    # Find the bounding box that encompasses both sets of points
    min_x = min(points1[:, 0].min(), points2[:, 0].min())
    max_x = max(points1[:, 0].max(), points2[:, 0].max())
    min_y = min(points1[:, 1].min(), points2[:, 1].min())
    max_y = max(points1[:, 1].max(), points2[:, 1].max())

    # Create grid based on the bounding box
    grid_x, grid_y = np.meshgrid(np.linspace(min_x, max_x, n), np.linspace(min_y, max_y, n))
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    def in_hull(p, hull):
        return all(np.dot(eq[:-1], p) + eq[-1] <= 0 for eq in hull.equations)

    # Check which points are in both hulls
    in_hull1_mask = np.apply_along_axis(in_hull, 1, grid_points, hull1)
    in_hull2_mask = np.apply_along_axis(in_hull, 1, grid_points, hull2)
    in_both_hulls_mask = in_hull1_mask & in_hull2_mask

    points_in_both_hulls = grid_points[in_both_hulls_mask]

    return points_in_both_hulls


def create_grid_in_convex_hull(points, n=100):
    hull = ConvexHull(points)
    grid_x, grid_y = np.meshgrid(np.linspace(0, 10, n), np.linspace(0, 10, n))
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    def in_hull(p_, hull_):
        return all(np.dot(eq[:-1], p_) + eq[-1] <= 0 for eq in hull_.equations)

    in_hull_mask = np.apply_along_axis(in_hull, 1, grid_points, hull)
    points_in_hull = grid_points[in_hull_mask]

    return points_in_hull


def unwarp_swc_only_z_dist(df_swc, gam_lower, df_lower_bvs, gam_upper, df_upper_bvs, pixel_size_um, z_step_um):
    # Compute the offsets relative to the lower band
    pxs, pys, pzs = df_swc.x.values, df_swc.y.values, df_swc.z.values
    pX = np.stack([pxs.flat, pys.flat]).T / pixel_size_um
    offsets_lower = gam_lower.predict(X=pX)
    deltas = offsets_lower
    deltas_um = deltas * z_step_um

    xy_shared = create_grid_in_convex_hulls(
        points1=df_lower_bvs[['x', 'y']].values, points2=df_upper_bvs[['x', 'y']].values)

    d_med_um = np.median(gam_upper.predict(X=xy_shared) - gam_lower.predict(X=xy_shared)) * z_step_um

    df_swc_flat = df_swc.copy()

    z_flat = pzs - deltas_um
    z_norm = z_flat / d_med_um

    df_swc_flat["z"] = z_norm

    return df_swc_flat, d_med_um
