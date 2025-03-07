import xarray
from data_containers import BehavioralData, BehavioralMLData
from data_util import invert_full_subject_code
from anatomy_util import load_contacts_pairs, get_elec_regions, group_elec_regions
EZZY_REGIONS_OF_INTEREST = ['IFG', 'MFG', 'SFG', #'OTHER_FRNT', 
                            'MTL', 'HPC', 'LTC', 
                            'IPC', 'SPC', 'OCC']
REGIONS_OF_INTEREST = ['FRNT', 'MTL', 'HPC', 'LTC', 'PAR', 'OCC']


def behavioral_data_groupby(subject_data:BehavioralData, by, session_factors=['subject', 'experiment', 'session']):
    subject_data = deepcopy(subject_data)
    if not isinstance(by, (str, int)):
        raise ValueError('behavioral_data_groupby() limited to grouping on a single factor of type str or int.')
    for group, events in subject_data.events.groupby(by):
        sessions = subject_data.sessions.merge(events[session_factors].drop_duplicates(session_factors),
                                               on=session_factors)
        if isinstance(subject_data, BehavioralMLData):
            mask = np.array(subject_data.events[by] == group)
            X = subject_data.X[mask]
            y = subject_data.y[mask] if not isinstance(subject_data.y, type(None)) else None
            groups = subject_data.groups[mask] if not isinstance(subject_data.groups, type(None)) else None
            sample_weight = subject_data.sample_weight[mask] if not isinstance(subject_data.sample_weight, type(None)) else None
            group_data = BehavioralMLData(sessions=sessions,
                                          events=events,
                                          X=X,
                                          y=y,
                                          groups=groups,
                                          sample_weight=sample_weight)
        elif isinstance(subject_data, BehavioralData):
            group_data = BehavioralData(sessions=sessions,
                                        events=events)
        else:
            raise ValueError
        yield group, group_data

# sub = 'R1328E_0_0'  #'R1001P_0_0'
# subject_data = cfr_enc[sub]

# session_number, session_data = next(behavioral_data_groupby(subject_data, by='session'))
# display(session_data.sessions)
# display(session_data.events.head(2))
# display(session_data.events.shape)
# print('session:', session_number)
# display(session_data.events.session.unique())
# display(session_data.X.shape)
# display(session_data.y.shape)
# display(session_data.groups.shape)
# display(session_data.sample_weight)


def load_subject_electrode_pairs(subject_data,
                                 region_col='major_region',
                                 regions_of_interest=REGIONS_OF_INTEREST):
    assert len(subject_data.sessions.subject.unique()) == 1
    subject, montage, localization = invert_full_subject_code(subject_data.sessions.subject.iloc[0], include_montage_localization=True)
    reader = CMLReader(subject=subject, montage=montage, localization=localization)

    _, pairs = load_contacts_pairs(reader, exclude_categories=list(), require_categories=False, suppress_cmlreaders_warnings=True)
    pairs = pairs.query(f'{region_col} in @regions_of_interest')
    return pairs


def average_over_regions(values,
                         channels,
                         channel_col='channel',
                         region_col='major_region',
                         regions_of_interest=REGIONS_OF_INTEREST):
    assert 'channel' in values.coords and 'frequency' in values.coords
    assert region_col in channels
    sizes = (len(values.frequency), len(regions_of_interest))
    all_region_values = xarray.DataArray(data=np.full(sizes, np.nan),
                                         coords={'frequency': values.frequency,
                                                 'region': regions_of_interest})
    for region, region_channels in channels.groupby(region_col):
        region_values = values.sel(channel=np.array(region_channels.label)).mean(channel_col, skipna=True)
        all_region_values[:, all_region_values.region==region] = region_values
    return all_region_values


def get_positive_negative_observations(subject_data):
    X = subject_data.X
    y = subject_data.y
    assert set(list(np.unique(y))) == {0, 1}
    X0 = X[y == 0]
    X1 = X[y == 1]
    return X1, X0


def get_effect(X0, X1, axis=0, statistic='hedges_g'):
    if statistic == 'hedges_g':
        from data_util import hedges_g
        effects = hedges_g(X0, X1, axis=axis)
    elif statistic == 't_stat':
        from scipy.stats import ttest_ind
        # effects = numpy2xarray_function(X1, X0, ).statistic
        effects = ttest_ind(X1, X0, axis=axis).statistic
        try:
            effects = xarray_drop_dimension(X0,
                                            dim_to_drop=X0.dims[axis],
                                            fill_values=effects)
            # print(effects.shape, effects.dims)
        except Exception as e:
            print('X1.shape', X1.shape, X1.dims)
            print('X0.shape', X0.shape, X0.dims)
            print(effects.size, effects.shape)
            raise e
            import pdb; pdb.set_trace()
    else:
        raise NotImplementedError
    return effects


def get_subject_region_frequency_standardized_effects(subject_data,
                                                      region_col='major_region',
                                                      regions_of_interest=REGIONS_OF_INTEREST,
                                                      contact_level_stat='hedges_g'):
    pairs = load_subject_electrode_pairs(subject_data,
                                         region_col=region_col,
                                         regions_of_interest=regions_of_interest)

    X1, X0 = get_positive_negative_observations(subject_data)
    effects = get_effect(X0, X1, axis=0, statistic=contact_level_stat)
    all_region_effects = average_over_regions(effects.unstack(),
                                              pairs,
                                              region_col=region_col,
                                              regions_of_interest=regions_of_interest)
    return all_region_effects


from typing import Callable

# extend numpy.ndarray-valued functions to return xarray.DataArrays
def numpy2xarray_function(arr:xarray.DataArray, dim:str, func:Callable, array_attributes=None, **kwargs):
    assert 'axis' not in kwargs, 'Use "dim" instead.'
    axis = list(arr.dims).index(dim)
    result = func(arr, axis=axis, **kwargs)
    if isinstance(array_attributes, list):
        for attribute in array_attributes:
            attribute_xarray = xarray_drop_dimension(arr, dim, fill_values=getattr(result, attribute))
            try:
                setattr(result, attribute, attribute_xarray)
            except AttributeError:
                result = get_mutable_instance(result)
                setattr(result, attribute, attribute_xarray)
    else:
        assert isinstance(array_attributes, type(None))
        result = xarray_drop_dimension(arr, dim, fill_values=result)

    return result


def xarray_drop_dimension(data_array, dim_to_drop, fill_values=None):
    """
    Create a new xarray.DataArray with a specified dimension dropped and filled with custom data or NaNs.

    Parameters:
    - data_array: xarray.DataArray
        The original DataArray to base the new DataArray on.
    - dim_to_drop: str
        The name of the dimension to drop.
    - fill_values: array-like, optional
        The data to fill the new DataArray with. If None, fills with NaNs.

    Returns:
    - xarray.DataArray
        A new DataArray with the specified dimension dropped and filled with the provided data or NaNs.
    """
    # Calculate the new shape after dropping the dimension
    new_shape = tuple(size for dim, size in zip(data_array.dims, data_array.shape) if dim != dim_to_drop)

    # Use fill_data if provided; otherwise, default to NaNs
    fill_values = fill_values if fill_values is not None else np.full(new_shape, np.nan)

    # Ensure the fill data matches the new shape
    if fill_values.shape != new_shape:
        raise ValueError("fill_values must match the shape of the DataArray after the dimension is dropped.")

    coords = {key: val if hasattr(val, 'coords') and key not in val.coords else np.array(val)
                for key, val in data_array.coords.items() if key != dim_to_drop}
    # print(data_array.dims)
    # print(coords.keys())
    # Create a new DataArray with the specified fill values
    new_data_array = xr.DataArray(
        fill_values,
        dims=[dim for dim in data_array.dims if dim != dim_to_drop],  # Keep only remaining dims
        coords={key: data_array.coords[key]  #if hasattr(val, 'coords') and key not in val.coords else np.array(val)
                for key in data_array.dims if key != dim_to_drop}  # Update coords
    )

    return new_data_array


from dataclasses import dataclass, fields, make_dataclass, field

def get_mutable_instance(immutable_instance):
    """
    Returns a new mutable class with the same attributes as the provided instance.

    Parameters:
    - immutable_instance: An instance of an immutable class (e.g., a namedtuple or frozen dataclass).

    Returns:
    - A new class instance that is a mutable version of the input instance's class.
    
    # Example usage:
    from collections import namedtuple

    # Immutable namedtuple
    Person = namedtuple("Person", ["name", "age", "statistics"])
    immutable_person = Person(name="Alice", age=30, statistics=np.array([1, 2, 3]))

    # Convert to a mutable version
    mutable_person = get_mutable_instance(immutable_person)
    mutable_person.statistics[0] = 99  # Now this is mutable
    print(mutable_person)     # Output: PersonMutable(name='Alice', age=30, statistics=array([99, 2, 3]))
    """

    original_class = type(immutable_instance)
    field_defs = []
    original_fields = [field for field in dir(immutable_instance)
                       if not (field.startswith('_') or isinstance(getattr(immutable_instance, field), Callable))]
    for f in (fields(original_class) if hasattr(original_class, '__dataclass_fields__') else original_fields):
        field_name = f.name if hasattr(f, 'name') else f
        field_type = f.type if hasattr(f, 'type') else type(getattr(immutable_instance, field_name))
        field_value = getattr(immutable_instance, field_name)
        
        # Use `default_factory` for mutable types
        if isinstance(field_value, (list, dict, np.ndarray)):
            field_defs.append((field_name, field_type, field(default_factory=lambda: field_value.copy()
                                                             if isinstance(field_value, np.ndarray) else field_type())))
        else:
            field_defs.append((field_name, field_type, field(default=field_value)))

    MutableClass = make_dataclass(original_class.__name__ + "Mutable", field_defs, frozen=False)
    return MutableClass(*[getattr(immutable_instance, field[0]) for field in field_defs])

import numpy as np
from skimage.measure import label
from scipy.stats import ttest_1samp

# python implementation of threshold-free cluster enhancement methods from EzzyEtal17 (/home2/yezzat/ye_code/Utilities/)

def tfce_compute(t):
    """
    Threshold-free cluster enhancement.

    Args:
      t: A NumPy array of t-statistics (or any statistical measure).

    Returns:
      A tuple containing:
        - tfce: The TFCE-enhanced t-statistic map.
        - tfce_pos: The TFCE-enhanced t-statistic map for positive values.
        - tfce_neg: The TFCE-enhanced t-statistic map for negative values.
    """
    pos_space = np.arange(0, np.max(t) + 0.05, 0.05)
    tfce_pos = np.zeros_like(t)

    for k in range(len(pos_space)): # k indexes the t-stat threshold
        h = t > pos_space[k]
        # h is a mask to indicate whether a t-statistic is above the current iteration's threshold
        # print(k)
        l = label(h, connectivity=2)  # 8-connectivity in MATLAB = 2 in skimage
        # l is a connectivity labeling
        # if time-frequency pixels are just 2 away, and are both above the threshold, they get labeled together
        # print(l)

        extent = np.zeros_like(l) # initialize extent, same shape as t-stats
        for z in range(1, np.max(l) + 1): # z indexes the connected components
            extent[l == z] = np.sum(l == z) # label the connected component with the number of pixels in that component
        # print(extent)
        # break
        
        for m in range(t.shape[0]): # m indexes dim1 of t-stats
            for n in range(t.shape[1]): # n indexes dim2 of t-stats
                tfce_pos[m, n] = tfce_pos[m, n] + extent[m, n]**0.5 * pos_space[k]**2
                # augment the 

    neg_space = np.arange(np.min(t), 0.00 + 0.05, 0.05)
    tfce_neg = np.zeros_like(t)

    for k in range(len(neg_space)):
        h = t < neg_space[k]
        l = label(h, connectivity=2)

        extent = np.zeros_like(l)
        for z in range(1, np.max(l) + 1):
            extent[l == z] = np.sum(l == z)

        for m in range(t.shape[0]):
            for n in range(t.shape[1]):
                tfce_neg[m, n] = tfce_neg[m, n] + extent[m, n]**0.5 * neg_space[k]**2

    tfce = tfce_pos + tfce_neg
    return tfce, tfce_pos, tfce_neg


def tfce_sign_resamp(y, its):
    """
    Compute the null t-distribution via sign flipping.

    Args:
      y: A NumPy array of data with shape (subjects, x, y).
      its: The number of iterations for sign-flipping.

    Returns:
      A tuple containing:
        - p_corr_tfce: TFCE-corrected p-values.
        - p_corr_t: T-statistic sign-flip permutation p-values.
        - tfce: The TFCE-enhanced t-statistic map.
        - tstat: The t-statistic map.
        - p_uncorr: Uncorrected p-values.
    """
    tfce_dist = np.full((its, y.shape[1], y.shape[2]), np.nan)
    tfce_pos_dist = np.full(its, np.nan)
    tfce_neg_dist = np.full(its, np.nan)
    t_pos_dist = np.full(its, np.nan)
    t_neg_dist = np.full(its, np.nan)

    for curIt in range(its):
        it_y = y.copy()
        rv = np.random.rand(y.shape[0])
        rv_idx = rv <= 0.5
        it_y[rv_idx] = -it_y[rv_idx]

        tstat, p_uncorr = ttest_1samp(it_y, 0, axis=0, nan_policy='omit')
        tmp = np.squeeze(tstat)

        if np.any(tmp > 0):
            t_pos_dist[curIt] = np.max(tmp[tmp > 0])
        else:
            t_pos_dist[curIt] = 0

        if np.any(tmp < 0):
            t_neg_dist[curIt] = np.abs(np.min(tmp[tmp < 0]))
        else:
            t_neg_dist[curIt] = 0

        tmp, tfce_pos, tfce_neg = tfce_compute(tmp)
        tfce_pos_dist[curIt] = np.max(tfce_pos)
        tfce_neg_dist[curIt] = np.max(tfce_neg)

        tfce_dist[curIt] = tmp

    tstat, p_uncorr = ttest_1samp(y, 0, axis=0, nan_policy='omit')
    if y.shape[1] != 1:
        tstat = np.squeeze(tstat)
    else:
        tstat = np.squeeze(tstat).T

    tfce, tfce_pos, tfce_neg = tfce_compute(tstat)

    p_corr_tfce = np.zeros_like(tstat)
    p_corr_t = np.zeros_like(tstat)

    for j in range(y.shape[1]):
        for k in range(y.shape[2]):
            if tstat[j, k] > 0:
                p_corr_tfce[j, k] = np.sum(tfce_pos[j, k] < tfce_pos_dist) / its
                p_corr_t[j, k] = np.sum(tstat[j, k] < t_pos_dist) / its
            else:
                p_corr_tfce[j, k] = np.sum(tfce_neg[j, k] < tfce_neg_dist) / its
                p_corr_t[j, k] = np.sum(tstat[j, k] < t_neg_dist) / its

    p_corr_tfce[p_corr_tfce == 0] = 1 / its
    p_corr_t[p_corr_t == 0] = 1 / its
    p_uncorr = np.squeeze(p_uncorr)

    return p_corr_tfce, p_corr_t, tfce, tstat, p_uncorr


def add_significance_to_imshow(p_values):
    for i in range(p_values.shape[0]):
        for j in range(p_values.shape[1]):
            if p_values[i, j] < 0.001:
                stars = '***'
            elif p_values[i, j] < 0.01:
                stars = '**'
            elif p_values[i, j] < 0.05:
                stars = '*'
            else:
                stars = ''

            text = plt.text(j, i, stars, ha="center", va="center", color="yellow")
