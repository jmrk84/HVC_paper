# -*- coding: utf-8 -*-
################################################################################
#  This file provides functions used to analze data for the following publication:
#  ...
#
#  (C) Copyright 2017
#  Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V.
#
#  HVC_RA_analysis.py is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License version 2 of
#  the License as published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#
################################################################################
"""
 Functions for the analysis of skeletons and synapses traced in the j2056 HVC dataset.
 Most functions take as parameter a path to a folder of annotation files or either an
 annotation file itself. Unpack the data zip file containing all annotation files
 and then adjust the paths accordingly when calling the functions.
 In case of questions or difficulties, please contact kornfeld@neuro.mpg.de
"""

__author__ = 'Joergen Kornfeld'

import os
import numpy as np
import collections
from matplotlib import pyplot as mplt
from knossos_utils import skeleton
from knossos_utils import skeleton_utils as su
from knossos_utils import skeleton_plotting as plt
from knossos_utils import synapses as sa
from knossos_utils import knossos_dataset as kds
from random import shuffle
from random import randint
import re
import glob
import networkx as nx
from compiler import ast
import copy
from collections import defaultdict
import pandas as pd
import scipy
import pandas
from scipy.interpolate import interp1d

try:
    from mayavi import mlab
except:
    print('Mayavi import issue. Is it installed? If yes, check your OpenGL driver for example.')

try:
    import pymc
except:
    print('Install the pymc package for Bayesian analysis functions')

def kde_of_axon_soma_dist_pdf(plot = False,
                              base_path='/axon analysis/bayesian/'):
    """
    Performs a Gaussian kernel density estimation of the measured LM data.
    :param plot: bool
    :return: scipy kde
    """

    # read LM axon data
    axonal_length_dist = pandas.read_excel(base_path + 'LM Axonal length.xlsx')
    LM_dists_axon_length = axonal_length_dist['radius']
    LM_amount_axon = axonal_length_dist['avg']

    re_sampled = []
    for d, amount in zip(LM_dists_axon_length, LM_amount_axon):
        re_sampled.extend([d]*int(amount))

    kde_pdf = scipy.stats.kde.gaussian_kde(re_sampled)

    if plot:
        x = np.linspace(1, 1500, 1000)
        mplt.figure()
        mplt.plot(x, kde_pdf(x), color='black')
        mplt.title('Gaussian KDE of LM axon length distribution')
        mplt.xlabel('soma dist [um]')
        mplt.ylabel('probability')

    return kde_pdf

def model_branch_density(plot = True,
                         base_path='/axon analysis/bayesian/'):
    """
    Bayesian model for the estimation of the soma distance based on the measured number of branch
    points, and the axonal length distribution as prior.

    :param plot: bool
    :param base_path: str
    :return:
    """

    # Convert measured LM axon path length to a gaussian kernel density estimated pdf
    dist_axon_pdf = kde_of_axon_soma_dist_pdf()

    # Fit an exponential to LM-branch density distribution
    a, b, c, = fit_LM_bd()

    # Read EM data
    data = pandas.read_excel(base_path + 'EM axon data.xlsx')
    num_branches = data['num main branches']
    branch_lengths = data['axon length main[um]']
    frac_INT_syns = data['fract INT']
    EM_axon_names = data['plot name']

    dist_distributions = []

    # Infer distance distributions
    def prob(dist, n, l):
        exp_lambda = LM_bd(dist, a, b, c) * l / 1000. # expected number of branches
        prob_val = dist_axon_pdf(dist) * (np.power(exp_lambda, n) * np.exp(- exp_lambda)) / np.math.factorial(n)
        return prob_val

    if plot:
        mplt.figure()

    for n_b, b_l in zip(num_branches, branch_lengths):
        # evaluate on 1000 soma distances
        x = np.linspace(0, 1500, 1000)
        y = []
        for x_i in x:
            y.extend(prob(x_i, n_b, b_l))
        dist_distributions.append((x, np.array(y)))

        # find median (0.5 quantile)
        x_med, y_med = find_q_quantile(0.5, x, np.array(y))

        if plot:
            mplt.plot(x, y, label='# branches {0:2d} length {1:2.1f} um'.format(n_b, b_l))

    if plot:
        mplt.legend(frameon=False)
        mplt.title('Posterior distributions')
        mplt.xlabel('soma distance [um]')
        mplt.ylabel('probability, not normalized')

        mplt.figure()

    est_distances = []
    errors_dist_left = []
    errors_dist_right = []
    for EM_axon_name, posterior in zip(EM_axon_names, dist_distributions):
        x_med, _ = find_q_quantile(0.5,  posterior[0], posterior[1])
        print('Median dist est of {0:s}: {1:3.3f})'.format(EM_axon_name, x_med))

        est_distances.append(x_med)

        # 0.15865 and 0.84135 for 1 SD
        # 0.0227 and 0.9772 for 2 SD
        x_left_eb, _ = find_q_quantile(0.0227, posterior[0], posterior[1])
        x_right_eb, _ = find_q_quantile(0.9772, posterior[0], posterior[1])
        print('0.0227 quantile of {0:s}: {1:3.3f})'.format(EM_axon_name, x_left_eb))
        print('0.9772 quantile of {0:s}: {1:3.3f})'.format(EM_axon_name, x_right_eb))
        errors_dist_left.append(x_left_eb)
        errors_dist_right.append(x_right_eb)

    if plot:
        mplt.errorbar(est_distances, frac_INT_syns, xerr=[errors_dist_left, errors_dist_right], fmt='o', color='black')

        for label, x, y in zip(EM_axon_names, est_distances, frac_INT_syns):
            mplt.annotate(
                label,
                xy = (x, y))

        mplt.title('Fract. int synapses vs bayesian distance estimates')
        mplt.xlim((-50., 1000.))
        mplt.ylim((-0.05, 0.8))
        mplt.xlabel('soma distance [um]')
        mplt.ylabel('fract. int. synapses outgoing')

    return dist_distributions

def find_q_quantile(q, x, y):
    """
    Manually finds quantiles on discrete data.

    :param q: float
    :param x: float
    :param y: float
    :return: float, float
    """

    tot_area = np.trapz(y, x)

    for idx in range(1, len(x)):
        if tot_area * q < np.trapz(y[0:idx], x[0:idx]):
            return x[idx], y[idx]

def fit_LM_bd(plot = False,
              base_path='/axon analysis/bayesian/'):
    """
    Fits an exponential to the LM measured branch density data. This is necessary,
    because the Bayesian posterior calculation needs to sample from continuous
    values. Returns the estimated parameters.

    :param base_path: str
    :return: float, float, float
    """

    # Read the LM data
    data = pandas.read_excel(base_path + 'branch_density_soma_distance_LM.xlsx')
    LM_dist = np.array(data['distance'])
    LM_bd_data = np.array(data['bd'])

    # Exponential
    def func(x, a, b, c):
        return a * np.exp(-b * x) + c

    # Initial values are provided, otherwise there are sometimes convergence problems in this case.
    popt, pcov = scipy.optimize.curve_fit(func, LM_dist, LM_bd_data, [50, 0.01, 1])
    a, b, c = popt

    if plot:
        mplt.figure()
        mplt.plot(LM_dist, LM_bd_data, '.', label='LM data')
        x = np.linspace(0, 1500, 1000)
        mplt.plot(x, func(x, a, b, c), color='black', label = 'y={0:2.3f}*e^(-{1:2.3f}*x)+{2:2.3f}'.format(a, b, c))
        mplt.title('LM distance vs branch density data look up')
        mplt.xlabel('soma distance [um]')
        mplt.ylabel('branch density 1/mm')
        mplt.legend(frameon=False)

    return a, b, c

def LM_bd(dist, a, b, c):
    return a * np.exp(-b * dist) + c


def load_HVC_RA_001_annotation(no_syn_distances=True,
                               path_to_annotation='/dendrite analysis/HVC_RA_001_only_dendrite.k.zip'):
    """
    Load Fig. 1 "synaptome" HVC(RA) dendrite with its synapses for analysis and plotting.
    :param no_syn_distances: bool, do not calculate synapse to soma distances (slow).
    :return: iterable of synapse annotation objects, skeleton annotation object
    """

    HVC_RA_001 = su.load_j0256_nml(path_to_annotation)
    HVC_syns, bad = sa.synapses_from_jk_anno(HVC_RA_001)
    soma_node = [n for n in HVC_RA_001[0].getNodes() if 'First' in n.getPureComment()][0]
    nxg = su.annoToNXGraph(HVC_RA_001)[0]

    if not no_syn_distances:
        # Set soma distance for each synapse
        for s in HVC_syns:
            try:
                s.soma_dist = nx.shortest_path_length(nxg, s.postNode, soma_node, weight='weight')/1000.
            except:
                print('Skipping synapse soma distance, no path to soma: ' + str(s.postNode))

    return HVC_syns, HVC_RA_001

def write_synapse_csv(syn_list, path_to_csv=''):
    """
    Writes a csv file with information about the passed iterable of Synapse objects.
    :param syn_list: iterable of Synapse objects
    :return:
    """

    btext =  open(path_to_csv, 'w')
    btext.write('Cell\ttype\taz length [nm]\tpre coord\tpost coord\tsoma dist [um]\n')
    for s in syn_list:
        s.type_tags = ['as']
        if s.type_tags:
            btext.write("{0}\t".format(s.source_annotation))
            btext.write("{0}\t".format(s.type_tags[0]))
            btext.write("{0:3.2f}\t".format(s.az_len))
            btext.write("{0},{1},{2}\t".format(s.postNode.getCoordinate()[0],
                                               s.postNode.getCoordinate()[1],
                                               s.postNode.getCoordinate()[2]))
            btext.write("{0},{1},{2}\t\n".format(s.preNode.getCoordinate()[0],
                                               s.preNode.getCoordinate()[1],
                                               s.preNode.getCoordinate()[2]))
            #btext.write("{0:3.2f}\t\n".format(s.soma_dist))
            #btext.write("{0}\n".format(s.postNode.annotation.filename))
    btext.close()

def estimate_HVC_RA_with_spine_path_length(path_to_folder=''):
    """
    Calculates path lengths of axons and dendrites separately.
    :param path_to_folder:
    :return:
    """

    files = glob.glob(path_to_folder)

    all_lengths_dendrite = []
    all_lengths_axon = []
    for f in files:
        annos = su.load_j0256_nml(f)
        dendrite = [a for a in annos if 'dendrite' in a.comment][0]
        try:
            axon = [a for a in annos if 'axon' in a.comment][0]
            all_lengths_axon.append(axon.physical_length() / 1000.)
        except:
            print('No axon found for cell {0}'.format(f))
        all_lengths_dendrite.append(dendrite.physical_length()/1000.)


    print('Total path length of all cells dendrites: {0}'.format(np.sum(all_lengths_dendrite)))
    print('Mean path length of all cells dendrites: {0}'.format(np.mean(all_lengths_dendrite)))
    print('Std path length of all cells dendrites: {0}'.format(np.std(all_lengths_dendrite)))


    print('Total path length of all cells axons: {0}'.format(np.sum(all_lengths_axon)))
    print('Mean path length of all cells axons: {0}'.format(np.mean(all_lengths_axon)))
    print('Std path length of all cells axons: {0}'.format(np.std(all_lengths_axon)))


    return


def estimate_HVC_RA_non_spine_path_length(path_to_folder='/dendrite analysis/dendritic path length without spines/*.zip'):
    """
    Calculate the dendritic EM path length for HVC(RA) reconstructions. This function further calculates the maximum
    extent of dendrites from soma-to-tip, for comparison with LM reconstructions.
    This function was not used for the calculation in the manuscript, but for an earlier version.

    :param path_to_folder: str, path to folder that contains the k.zip annotations
    :return:
    """

    files = glob.glob(path_to_folder)
    total_len = 0.
    all_tip_lengths = []
    all_dendrites_lengths_without_spines = []
    for f in files:
        annos = su.load_j0256_nml(f)
        dendrite = [a for a in annos if 'dendrite' in a.comment][0]
        path_length = dendrite.physical_length()/1000.
        total_len += path_length
        all_dendrites_lengths_without_spines.append(path_length)

        # get maximum soma-dendrite tip distance

        # find soma node
        soma_node = [n for n in dendrite.getNodes() if 'soma' in n.getPureComment()][0]

        # make a list of all end nodes in dendrite
        nxg = su.annoToNXGraph(dendrite)[0]
        end_nodes = list({k for k, v in nxg.degree().iteritems() if v == 1})

        # calc dist to soma node of all end nodes
        soma_tip_dists = [soma_node.distance_scaled(en) for en in end_nodes]
        soma_tip_dists.sort(reverse=True)
        all_tip_lengths.append(soma_tip_dists[0])

        print('Cell: {0} Path length: {1}; max tip dist: {2}'.format(dendrite.filename, path_length,
                                                                     soma_tip_dists[0]/1000.))
    print('Mean tip length: {0}'.format(np.mean(all_tip_lengths)/1000.))
    print('Std tip length: {0}'.format(np.std(all_tip_lengths)/1000.))
    print('Total path length of all cells: {0}'.format(np.sum(all_dendrites_lengths_without_spines)))
    print('Mean path length of all cells: {0}'.format(np.mean(all_dendrites_lengths_without_spines)))
    print('Std path length of all cell: {0}'.format(np.std(all_dendrites_lengths_without_spines)))

    return

def mcmc_synaptic_area(syns_observed, axon_length, avg_area):
    """
    Bayesian error model for synapses on axons. Was not used for the manuscript.
    :param syns_observed: iterable
    :param axon_length: iterable
    :return:
    """

    syn_dens = pymc.Uniform('syn_dens', 0, 200.) # prior for synapse densities
    len_axon = pymc.Uniform('len_axon', 0., 10., value=[axon_length], observed=True) # observed length

    @pymc.deterministic
    def exp_syns(syn_dens=syn_dens, len_axon=len_axon):
        return syn_dens * len_axon

    poiss = pymc.Poisson('poiss', mu=exp_syns, value=[syns_observed], observed=True)

    @pymc.deterministic
    def syn_area(syn_dens=syn_dens, avg_area=avg_area, len_axon=len_axon):
        return syn_dens * len_axon * avg_area / len_axon

    # run mc sampling
    model = pymc.MCMC([poiss, syn_dens, len_axon, syn_area])
    model.sample(20000, burn=10000)
    print(model.stats())

    pymc.Matplot.plot(model)
    pymc.graph.dag(model)
    return


def mcmc_poisson_synapse_density(syns_observed, axon_length):
    """
    Bayesian error model for synapses on axons. Was not used for the manuscript.
    :param syns_observed: iterable
    :param axon_length: iterable
    :return:
    """

    syn_dens = pymc.Uniform('syn_dens', 0, 100.) # prior for synapse densities
    len_axon = pymc.Uniform('len_axon', 0., 10., value=[axon_length], observed=True) # observed length

    @pymc.deterministic
    def exp_syns(syn_dens=syn_dens, len_axon=len_axon):
        return syn_dens * len_axon

    poiss = pymc.Poisson('poiss', mu=exp_syns, value=[syns_observed], observed=True)

    # run mc sampling
    model = pymc.MCMC([poiss, syn_dens, len_axon])
    model.sample(20000, burn=10000)

    # get highest posterior density interval
    print(model.stats())

    pymc.Matplot.plot(model)
    return


def mcmc_binomial_synapse_fraction(n_total_syns, hit_syns):
    """
    Bayesian error model for synapses on axons. Was not used for the manuscript.
    :param n_total_syns: iterable
    :param hit_syns: iterable
    :return:
    """

    # use conjugate prior for Binomial
    # p_HIT = pymc.Beta('p_HIT', 10, 10)
    p_HIT = pymc.Uniform('p_HIT', 0.0, 1.0)
    N_HIT = pymc.Binomial('N_HIT', n=n_total_syns, p=p_HIT, value=hit_syns, observed=True)
    model = pymc.MCMC([p_HIT, N_HIT])

    model.sample(20000, burn=10000)

    print(model.stats())

    pymc.Matplot.plot(model)
    # get quantiles
    median = model.stats()['p_HIT']
    return (median, q1, q2)

def estimate_labeling_efficiency_soma(path_to_annotation='/soma analysis/soma_map.k.zip'):
    """
    Analysis of somatic labeling efficiency.

    :param path_to_annotation: str, path to soma annotation file
    :return:
    """
    soma_annotation = su.load_j0256_nml(path_to_annotation)

    nodes = soma_annotation[0].getNodes()
    n_total = len(nodes)
    n_RA = len([n for n in nodes if 'black' in n.getPureComment()])
    n_pInt = len([n for n in nodes if 'INT' in n.getPureComment()])
    n_pRA = len([n for n in nodes if 'RA' in n.getPureComment()])
    n_pX = len([n for n in nodes if 'X' in n.getPureComment()])
    n_uncl = len([n for n in nodes if 'Unclear' in n.getPureComment()])
    n_neuron = len([n for n in nodes if len(n.getPureComment()) > 1])
    n_glia = len([n for n in nodes if len(n.getPureComment()) == 0])

    fraction_RA_total = float(n_RA) / n_neuron
    fraction_RA_neuron_int = float(n_RA) / (n_neuron-n_pInt)
    fraction_RA_pRA = float(n_RA) / (n_RA+n_pRA)

    print('Total soma: {0}'.format(n_total))
    print('neuron: {0}'.format(n_neuron))
    print('non neuron: {0}'.format(n_glia))
    print('RA: {0}'.format(n_RA))
    print('pRA: {0}'.format(n_pRA))
    print('pINT: {0}'.format(n_pInt))
    print('pX: {0}'.format(n_pX))
    print('unclear: {0}'.format(n_uncl))
    print('RA/neuron = {0:0.2f}'.format(fraction_RA_total))
    print('RA/(neuron-pINT) = {0:0.2f}'.format(fraction_RA_neuron_int))
    print('RA/(RA+pRA) = {0:0.2f}'.format(fraction_RA_pRA))

    return


def find_missing_tasks(dataset_scale = (11.,11.,29.),
                       csv_path = '*.csv',
                       consensi_path = '',
                       redundant_anno_path = ''):
    """
    This function reads a heidelbrain task file with seed coords and then analyzes consensus files given in a folder.
    Seeds that do not have a consensus file are reported and can then be reseeded.

    :param csv_path:
    :param consensi_path:
    :return:
    """

    # parse csv file that contains the seed coords
    get_coord = lambda x: map(int, re.search(r'^.*\t.*\t(?P<coord>\d*\t\d*\t\d*)\t\d*$', x).group('coord').split())

    with open(csv_path, 'r') as csv_task_file:
        # do cheap regex parsing
        tasks = csv_task_file.readlines()
        coords = [get_coord(line) for line in tasks]

    # load all consensi in a folder
    annotation_files = glob.glob(consensi_path + '*.k.zip')
    consensi = []
    for af in annotation_files:
        anno = su.load_j0256_nml(af)
        if len(anno) > 1:
            raise Exception('Consensus file ' + af + ' has more than one skeleton tree.')
        consensi.extend(anno)
    # make a kdtree of consensi
    c_tree = su.annosToKDtree(consensi)

    # load all redundant annotations in a folder
    annotation_files = glob.glob(redundant_anno_path + '*.k.zip')
    redundant = []
    for af in annotation_files:
        anno = su.load_j0256_nml(af)
        if len(anno) > 1:
            raise Exception('Redundant file ' + af + ' has more than one skeleton tree.')
        redundant.extend(anno)
    # make a kdtree of consensi
    r_tree = su.annosToKDtree(redundant)

    missing_seed_coords_cons = []
    missing_seed_coords_red = []
    missing_seed_coords_red_all = []
    # run all seed coords against the kdtree with minimal radius
    # scale coords
    scaled_coords = np.multiply(np.array(coords), np.array(dataset_scale))

    hits = []

    for coord, task_line in zip(scaled_coords, tasks):

        hits_red = r_tree.query_ball_point(coord, 50.)
        hits_red = len(ast.flatten(hits_red))
        hits_cons = c_tree.query_ball_point(coord, 50.)
        hits_cons = len(ast.flatten(hits_cons))
        hits.append((coord, hits_red, hits_cons))

        if hits_red == 2 and hits_cons == 0:
            missing_seed_coords_cons.append(coord)
            print(task_line)
        if hits_red == 1 and hits_cons == 0:
            missing_seed_coords_red.append(coord)
        if hits_red == 0 and hits_cons == 0:
            missing_seed_coords_red_all.append(coord)

    # unscale
    missing_seed_coords_cons = np.divide(np.array(missing_seed_coords_cons), np.array(dataset_scale)).astype(np.uint32)
    missing_seed_coords_red = np.divide(np.array(missing_seed_coords_red), np.array(dataset_scale)).astype(np.uint32)

    return missing_seed_coords_cons, missing_seed_coords_red, missing_seed_coords_red_all


def find_contact_sites(presynaptic_annos,
                       postsynaptic_annos,
                       spotlight_radius = 5000.):
    """
    Finds node-pairs between different annotations with maximum distance
    spotlightRadius. Only looks into the pre -> post direction.  Not used for manuscript.
    """

    # contact_sites is a list of lists, where each list contains:
    # 0: node in presynaptic anno
    # 1: node in postsynaptic anno
    # 2: euclidian dist of the two in nm

    presynaptic_annos = copy.copy(presynaptic_annos)
    postsynaptic_annos = copy.copy(postsynaptic_annos)

    contact_sites = []

    post_tree_kds = []
    for post_anno in postsynaptic_annos:
        post_tree_kds.append(su.annoToKDtree(post_anno)[0])

    for pre_anno in presynaptic_annos:
        post_trees = su.annosToKDtree(postsynaptic_annos)

        nodes = list(pre_anno.getNodes())
        coords = [node.getCoordinate_scaled() for node in nodes]
        # query each node separately
        for n in nodes:
            hits = post_trees.query_ball_point(n.getCoordinate_scaled(), spotlight_radius)
            # if there are not hits, this node can safely be ignored for further analysis
            if not hits:
                pre_anno.removeNode(n)

        nx_a = su.annoToNXGraph(pre_anno)[0]
        for cc in nx.connected_components(nx_a):
            # each cc is a set of nodes
            nodes = list(cc)
            coords = [n.getCoordinate_scaled() for n in nodes]
            # test all ccs against all postsynaptic annos
            for post_kd in post_tree_kds:
                found_post_nodes, dists = post_kd.query_k_nearest(coords, k=1, return_dists=True)
                triplets = zip(nodes, found_post_nodes, dists)
                triplets = [t for t in triplets if t[2] < spotlight_radius]
                if triplets:
                    triplets.sort(key = lambda x: x[2])
                    # only the triplet with the shortest distance should be kept
                    contact_sites.append(triplets[0])
    return contact_sites

def chunks(l, n):
    """
    Yield successive chunks of size n from iterable l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def make_neurite_touch_task_files(path_to_out_folder = '',
                                  task_prefix='touch_task_',
                                  tagged_consensi_path='',
                                  scaling=(11.,11.,29.),
                                  cs_id_offset = 0):


    """
    Creates task files, each containing 100 neurite proxmities to test.
    Not used for manuscript.
    :return:
    """

    # load annotation files
    #int_axons, rap_dendrites = load_int_axons_HVC_RA_dendrites(tagged_consensi_path)
    annos = su.load_j0126_skeletons_from_folder(tagged_consensi_path)

    shuffle(annos)
    # find putative contact sites
    contact_sites = find_contact_sites(annos[0:100], annos[101:200], spotlight_radius=2500.)
    #contact_sites = find_contact_sites(int_axons, rap_dendrites)

    shuffle(contact_sites)
    contact_sites = contact_sites[0:2000]

    split_cs = chunks(contact_sites, 50)
    # each task file contains x putative touches that should be analyzed
    # for cs in contact_sites:

    # split up contact_sites into task files with 50 each

    cs_id = cs_id_offset

    for task_cnt, task_cs in enumerate(split_cs):
        skel_obj = skeleton.Skeleton()
        anno = skeleton.SkeletonAnnotation()
        anno.scaling = scaling

        for cs in task_cs:
            skel_obj.add_annotation(anno)

            # create 3 nodes from scratch
            # a----b----c: where b is the "decision" node that contains a todo comment
            # the old node object instances can potentially be reused.
            cs_id += 1
            a = skeleton.SkeletonNode()
            a_coord = cs[0].getCoordinate()
            c_coord = cs[1].getCoordinate()
            a.from_scratch(anno, a_coord[0], a_coord[1], a_coord[2])
            a.setPureComment(str(cs_id))
            anno.addNode(a)

            todo_coord = ((np.array(a_coord)+np.array(c_coord))/2.).astype(np.int)
            b = skeleton.SkeletonNode()
            b.from_scratch(anno, int(todo_coord[0]), int(todo_coord[1]), int(todo_coord[2]))
            b.setPureComment('todo')
            anno.addNode(b)
            a.addChild(b)

            c = skeleton.SkeletonNode()
            c_coord = cs[1].getCoordinate()
            c.from_scratch(anno, c_coord[0], c_coord[1], c_coord[2])
            c.setPureComment(str(cs_id))
            anno.addNode(c)
            b.addChild(c)
            outfile = path_to_out_folder + task_prefix + str(task_cnt) + '.k.zip'

        skel_obj.to_kzip(outfile)
        print('Writing ' + outfile)

    return

def convert_coordinates_to_nodes(coords, comments, path = '.k.zip'):
    """
    Helper function to write a kzip that has nodes at the passed coordinates

    :param coords: iterable
    :param comments: str
    :param path: str
    :return:
    """

    skel_obj = skeleton.Skeleton()
    anno = skeleton.SkeletonAnnotation()

    anno.scaling = (11.,11.,29.)
    for cur_coord_list, cur_comment in zip(coords, comments):
        for c in cur_coord_list:
            node = skeleton.SkeletonNode()
            node.from_scratch(anno, c[0], c[1], c[2])
            node.setPureComment(cur_comment)
            anno.addNode(node)
    skel_obj.add_annotation(anno)
    skel_obj.to_kzip(path)

    return


def load_int_axons_HVC_RA_dendrites(tagged_consensi_path=''):
    """
    Load the interneuron annotation files and the HVC(RA) annotation files and return the axon and dendrite as
    annotation objects. Not used for manuscript.

    :return:
    """

    # load all kzips in source folder
    int_axons = []
    int = []

    rap_dendrites= []
    rap = []

    annotation_files = glob.glob(tagged_consensi_path + '*.k.zip')

    for af in annotation_files:
        print('Loading file ' + af)
        annos = su.load_j0256_nml(af)
        int = False
        rap = False
        for a in annos:
            if [n for n in a.getNodes() if 'int' in n.getPureComment()]:
                int = True
            elif [n for n in a.getNodes() if 'rap' in n.getPureComment()]:
                rap = True

        for a in annos:
            if [n for n in a.getNodes() if 'xon' in n.getPureComment()]:
                if int:
                    int_axons.append(a)
            elif [n for n in a.getNodes() if 'endrite' in n.getPureComment()]:
                if rap:
                    rap_dendrites.append(a)

    return int_axons, rap_dendrites


def make_end_node_classification_task(path_to_in_folder = '/dendrite analysis/dendritic path length without spines/*.zip',
                                      path_to_out_folder = ''):
    """
    Creates a task file for the manual classification of whether a dendritic branch ends inside the dataset or is
    severed.

    :param path_to_in_folder: str
    :param path_to_out_folder: str
    :return:
    """

    files = glob.glob(path_to_in_folder)
    for f in files:
        annos = su.load_j0256_nml(f)
        dendrite = [a for a in annos if 'dendrite' in a.comment][0]
        # make a list of all end nodes in dendrite
        nxg = su.annoToNXGraph(dendrite)[0]
        end_nodes = list({k for k, v in nxg.degree().iteritems() if v == 1})
        [en.setPureComment('todo') for en in end_nodes]

        skel_obj = skeleton.NewSkeleton()
        skel_obj.add_annotation(dendrite)
        skel_obj.to_kzip(path_to_out_folder + os.path.basename(f))

    return


def read_end_node_classification_task(path_to_in_folder = '/dendrite analysis/dendritic end node analysis/*.zip'):
    """
    Analyzes the distances from annotated dendritic terminations to the soma and
    creates an excel file with the results.

    :param path_to_in_folder: str
    :return:
    """

    natural_dataset_fraction = []
    all_terminal_nodes = []
    all_dataset_nodes = []

    files = glob.glob(path_to_in_folder)
    for f in files:
        cell = su.load_j0256_nml(f)[0]

        # get soma node
        try:
            soma_node = [n for n in cell.getNodes() if 'soma' in n.getPureComment()][0]
        except:
            raise Exception('No soma node found in {0}'.format(f))

        # get annotated dendritic terminations
        terminal_nodes = [n for n in cell.getNodes() if 'termination' in n.getPureComment()]
        dataset_end_nodes = [n for n in cell.getNodes() if 'dataset' in n.getPureComment()]

        all_dataset_nodes.extend(dataset_end_nodes)
        all_terminal_nodes.extend(terminal_nodes)

        natural_dataset_fraction.append(len(dataset_end_nodes) / float(len(terminal_nodes)))
        print('{0} num termination: {1} num dataset severed: {2}'.format(cell.filename,
                                                                         len(terminal_nodes),
                                                                         len(dataset_end_nodes)))

    print('total, natural {0}, dataset {1}'.format(len(all_dataset_nodes), len(all_terminal_nodes)))
    # fraction of natural ends vs dataset
    print('Mean dataset/natural: {0}, {1} SD'.format(np.mean(natural_dataset_fraction), np.std(natural_dataset_fraction)))
    print('percent severed: {0}%'.format(float(len(all_terminal_nodes)) / (len(all_terminal_nodes) + len(all_dataset_nodes))*100.))

    return

def make_synapse_cnt_task_files(coord_chunk_size = 200,
                                out_folder='',
                                file_prefix='coord_chunks_',
                                bbone_synapse_annotation_file=''):
    """
    Generates .k.zip containers with individual nodes at locations where synapses should be counted for the spine
    counting task.

    :param coord_chunk_size: int
    :param out_folder: str
    :param file_prefix: str
    :return:
    """

    bbone_synapses_annos = su.load_j0256_nml(bbone_synapse_annotation_file)

    post_dendrites = [a for a in bbone_synapses_annos if 'post' in a.comment]

    end_nodes = []
    for post_dendrite in post_dendrites:
        nxg = su.annoToNXGraph(post_dendrite)[0]
        end_nodes.extend(list({k for k, v in nxg.degree().iteritems() if v == 1}))

    coords = [n.getCoordinate() for n in end_nodes]

    print('Found in total {num_end_nodes} end nodes.'.format(num_end_nodes=len(coords)))
    # make sure that annotators do not see coords that had some previous order
    shuffle(coords)
    chunked = chunks(coords, coord_chunk_size)
    for chunk_cnt, chunk_coords in enumerate(chunked):
        # create NewSkeleton object
        skel_obj = skeleton.NewSkeleton()
        anno = skeleton.SkeletonAnnotation()
        anno.scaling=(11.,11.,29.)
        skel_obj.add_annotation(anno)

        # create nodes
        for coord in chunk_coords:
            node_obj = skeleton.SkeletonNode()
            node_obj.from_scratch(anno, coord[0], coord[1], coord[2])
            node_obj.setPureComment('todo')
            anno.addNode(node_obj)
        skel_obj.to_kzip(out_folder+file_prefix+str(chunk_cnt)+'.k.zip')
    return

def make_known_axon_as_sy_task(path_to_axon_src_kzip = '/dendrite analysis/type classification/'
                                                       'combined_as_sy_axons_for_type_classification.001.k.zip',
                               path_to_task_kzip = '.k.zip'):
    """
    Creates a task file for the blind synapse type annotation from
    axons of identified type.
    Each skeleton node with comment "type" is presented to the annotator
    as a single presynaptic location for classification.
    Classification is performed by adding "-as" or "-sy" to the type comment.
    :return:
    """

    # The tree comments in the path_to_axon_src_kzip encode the identity
    # of the cell
    annos = su.load_j0256_nml(path_to_axon_src_kzip)
    type_nodes = []
    for anno in annos:
        type_nodes.extend([n for n in anno.getNodes()
                           if 'type' in n.getPureComment()])

    skel_obj = skeleton.NewSkeleton()
    anno = skeleton.SkeletonAnnotation()
    anno.scaling = (11.,11.,29.)
    skel_obj.add_annotation(anno)

    for type_node in type_nodes:
        type_node.setPureComment('type')
        anno.addNode(type_node)
    skel_obj.to_kzip(path_to_task_kzip)

    return

def analyze_known_axon_as_sy_task(path_to_axon_src_kzip = '/dendrite analysis/type classification/'
                                                       'combined_as_sy_axons_for_type_classification.001.k.zip',
                                  path_to_annotated_kzip = '/dendrite analysis/'
                                                           'type classification/synapse_type_analysis.022.k.zip'):
    """
    Analyzes the result of the annotated task file for the blind
    synapse type annotation.

    :return:
    """
    # Must be single tree
    type_anno = su.load_j0256_nml(path_to_annotated_kzip)[0]

    gt_type_annos = su.load_j0256_nml(path_to_axon_src_kzip)
    gt_sy_coords = set()
    gt_as_coords = set()
    for gt_type_anno in gt_type_annos:
        if 'inhibitory' in gt_type_anno.comment:
            # the currently parsed skeleton tree belongs to a neuron
            # that makes symmetric synapses (i.e. aspiny)
            gt_sy_coords |= set([tuple(n.getCoordinate()) for n in gt_type_anno.getNodes()])
        elif 'excitatory' in gt_type_anno.comment:
            gt_as_coords |= set([tuple(n.getCoordinate()) for n in gt_type_anno.getNodes()])

    annotated_types = [(tuple(n.getCoordinate()), n.getPureComment()) for n in type_anno.getNodes()
                       if 'type' in n.getPureComment()]

    correct_classified = []
    incorrect_classified = []

    for a_coord, a_type in annotated_types:
        if a_coord in gt_sy_coords and 'sy' in a_type:
            correct_classified.append(a_coord)
        elif a_coord in gt_as_coords and 'as' in a_type:
            correct_classified.append(a_coord)
        else:
            incorrect_classified.append(a_coord)

    print('Number correct: {0}'.format(len(correct_classified)))

    return

def make_consensus_task_files(path_to_in_folder = '',
                              path_to_out_folder = '',
                              task_prefix='consensus_iter_2_'):
    """
    Reads all skeleton annotation files from a folder, matches them through a kd-tree and then creates
    consensus task files.
    :param: path_to_folder, str
    :return:
    """

    get_task_annotator = lambda x: re.search(r'^.*-(?P<taskperson>.*-.*)-\d{8}-\d{6}-final.*$', x.filename).group('taskperson')

    annotation_files = glob.glob(path_to_in_folder + '*.k.zip')
    annotations = []
    for af in annotation_files:
        anno = su.load_j0256_nml(af)
        if len(anno) > 1:
            raise Exception('Annotation file ' + af + ' has more than one skeleton tree.')
        annotations.extend(anno)
    reseeds = []
    matched_skeletons, _ = su.annotation_matcher(annotations, spotlightRadius=400., visual_inspection=False)
    for task_cnt, same_cells in enumerate(matched_skeletons):
        if len(same_cells) < 2:
            reseeds.extend([n.getCoordinate() for n in same_cells[0].getNodes() if 'First' in n.getPureComment()])
            print('Reseed task ' + get_task_annotator(same_cells[0]))
            continue
        skel_obj = skeleton.NewSkeleton()
        for num, cell in enumerate(same_cells):
            skel_obj.add_annotation(cell)
            cell.setComment(get_task_annotator(cell))
        outfile = path_to_out_folder + task_prefix + str(task_cnt) + '.k.zip'
        skel_obj.to_kzip(outfile)

        print('Writing ' + outfile)

    return reseeds, matched_skeletons

def read_neurite_touch_task_files(path_to_folder=''):
    """
    Parse results from neurite touch task files. At least two annotators need to agree on a touch to confirm it.
    Not used for manuscript.

    :param path_to_folder:
    :return: iterable of found touches
    """

    def is_int(s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    annotation_files = glob.glob(path_to_folder + '*.k.zip')
    annotations = []
    for af in annotation_files:
        annotations.append(su.load_j0256_nml(af)[0])

    # lump all redundant touches together
    nxg = su.annoToNXGraph(annotations, merge_annotations_to_single_graph=True)

    # cc analysis
    ccs = list(nx.connected_component_subgraphs(nxg, copy=False))

    match_dict = collections.defaultdict(list)
    for cc in ccs:

        # get number
        this_ID = set([int(n.getPureComment()) for n in cc.nodes() if is_int(n.getPureComment())]).pop()
        print this_ID
        # get decision
        touch_decision = set([int(n.getPureComment()) for n in cc.nodes() if is_int(n.getPureComment())]).pop()
        node_coords = [n.getCoordinate() for n in cc.nodes() if is_int(n.getPureComment())]
        if True in map(lambda x: 'ouch' in x, [n.getPureComment() for n in cc.nodes()]):
            match_dict[this_ID].append((True, node_coords))
        elif True in map(lambda x: 'no' in x, [n.getPureComment() for n in cc.nodes()]):
            match_dict[this_ID].append((False, node_coords))

    found_touches = []
    # vote on touches
    for this_ID, v in match_dict.items():
        if len(v) < 3:
            print('Redundancy not reached')
        else:
            if len([el for el in v if el[0]]) > 2:
                # it shouldn't matter which node coords to use further
                found_touches.append((v[0][1]))
    return found_touches

def found_touches_to_kzip(found_touches,
                          path_to_out_file = '.k.zip',
                          scaling = (11.,11.,29.)):
    """
    Convert found touches to an analyzable kzip container for opening in KNOSSOS. Not used for manuscript.

    :param found_touches: iterable
    :param path_to_out_file: str
    :param scaling: 3-tuple of floats
    :return:
    """

    skel_obj = skeleton.Skeleton()
    anno = skeleton.SkeletonAnnotation()
    anno.scaling = scaling
    skel_obj.add_annotation(anno)

    for num, touch in enumerate(found_touches):
        # create 2 nodes from scratch:
        # a----b
        a = skeleton.SkeletonNode()
        a.from_scratch(anno, touch[0][0], touch[0][1], touch[0][2])
        a.setPureComment(str(num))
        anno.addNode(a)

        b = skeleton.SkeletonNode()
        b.from_scratch(anno, touch[1][0], touch[1][1], touch[1][2])
        b.setPureComment('todo')
        anno.addNode(b)

        a.addChild(b)
    skel_obj.to_kzip(path_to_out_file)
    return

def load_direct_HVC_RA_HVC_RA_synapses(synapse_path = '/dendrite analysis/homotypic synapses/*.zip',
                                       cell_path = '/dendrite analysis/axon dendrite separated/'):
    """
    Load double labeled (i.e. homotypic) synapse annotations.

    :return: iterable of synapse objects, iterable of syn_soma_distances,
    iterable of skeleton annotation objects, iterable of all synapses found
    """

    files_synapses = glob.glob(synapse_path)

    cells = []
    syns_confirmed_soma_dist = []
    total_path = 0.
    all_syns = []
    all_confirmed_synapses = []

    for f_syns in files_synapses:
        # this is a bit of a hack - the synapse files and the files in this folder must have the same name
        annos = su.load_j0256_nml(cell_path + os.path.basename(f_syns))
        anno_s = su.load_j0256_nml(f_syns)[0]

        for a in annos:
            if a.comment == 'dendrite':
                anno = a
        print anno.filename

        # calculate dendritic length and axonic length
        anno.soma_node = [n for n in anno.getNodes() if 'First Node' in
                          n.getPureComment()][0]
        anno_s.soma_node = [n for n in anno_s.getNodes() if 'First Node' in
                          n.getPureComment()][0]

        nxg = su.annoToNXGraph(anno_s)[0]

        # extract synapses
        this_syns = sa.synapses_from_jk_anno(anno_s)[0]

        confirmed_synapses = [s.preNode for s in this_syns if 'confirmed' in s.tags]
        all_confirmed_synapses.extend(confirmed_synapses)
        all_syns.append(this_syns)

        for s in confirmed_synapses:
            try:
                s.soma_dist = nx.shortest_path_length(nxg, s.postNode, anno_s.soma_node,\
                                                  weight='weight')/1000.
                syns_confirmed_soma_dist.append(s)
            except:
                print('No link between soma and post node')
                s.soma_dist = 0.

        this_cell_len = anno.physical_length()/1e6
        print('Path length: {0:3.3f}'.format(this_cell_len))
        total_path += this_cell_len
        cells.append(anno)
    print('Total path length: {0:3.3f}'.format(total_path))

    return all_confirmed_synapses, syns_confirmed_soma_dist, cells, all_syns


def get_spine_count_data(path = '/axon analysis/transsynaptic tracing/.../'):
    """
    Parses the results of the manual synapses per spine count annotations.

    :return: dict of redundant counts, kdtree with counts
    """

    files = glob.glob(path + '*.k.zip')
    count_annos = [su.load_j0256_nml(f)[0] for f in files]
    print('All {0} spine synapse cnt result file loaded.'.format(len(count_annos)))
    #print count_annos
    # create one big kd-tree with all spine counts!
    all_counts = []
    all_coords = []
    insufficient_counts = []
    all_pairs = []
    for a in count_annos:
        pairs = [(n.getCoordinate_scaled(), int(re.sub(r"\D", "", n.getPureComment()))) for n in a.getNodes()
            if re.search(r'\d', n.getPureComment())]
        all_pairs.extend(pairs)

    #red_dict = dict([(tuple(p[0]), p[1]) for  p in all_pairs])
    red_dict = {}
    for p in all_pairs:
        # match results belonging to the same spine end node together
        try:
            red_dict[tuple(p[0])].append(p[1])
        except:
            red_dict[tuple(p[0])] = [p[1]]

    # find spines that have insufficient count to give them out again
    for coord in red_dict.keys():
        if len(red_dict[coord]) < 2:
            print('Insufficient cnt')
            insufficient_counts.append((int(coord[0]/11.), int(coord[1]/11.), int(coord[2]/29.)))
        red_dict[coord] = np.mean(red_dict[coord])

    coords, cnt = zip(*red_dict.iteritems())
    spine_kdtree = su.KDtree(cnt, coords)

    return red_dict, spine_kdtree


def post_HVC_RA_random_sample_analysis(syn_cnt_results_path = '/axon analysis/transsynaptic tracing/from box sampling/synapse counting task/',
                                       excel_out_path='X:/j0256_analysis/random sampling HVC RA post tracing analysis/',
                                       bbone_synapse_annotation_file='/axon analysis/transsynaptic tracing/from box sampling/orphans.635.k.zip'):

    """
    Performs the same postsynaptic type classification analysis as the one for the fully traced HVC(RA) axons (see
    post_HVC_RA_axon_analysis()). This can be seen as a cleaned-up version of post_HVC_RA_axon_analysis().

    :return:
    """

    # Load spine synapse counting results
    _, spine_kdtree = get_spine_count_data(syn_cnt_results_path)

    # Load file with dendrites that contain the annotation of whether to exclude a synapse / dendrite and which
    # dendritic stretch to consider for the spine analysis
    bbone_synapses_annos = su.load_j0256_nml(bbone_synapse_annotation_file)

    post_dendrites = [a for a in bbone_synapses_annos if 'post' in a.comment]
    print('Number post dendrites found: {0}'.format(len(post_dendrites)))

    # After this, each dendrite_anno will contain its classification
    for dendrite_anno in post_dendrites:
        dendrite_anno = classify_post_dendrite(spine_kdtree, dendrite_anno)
        if dendrite_anno.bb_len < 10.:
            print('Backbone shorter than 10 um')

    # calculate inter-synapse distance from labeled synapses
    axons = [a for a in bbone_synapses_annos if 'Axon' in a.comment]
    print('Number axons found: {0}'.format(len(axons)))

    syn_densities = get_synapse_densities_for_axons(axons)
    print('Total mean synapse density: {0}, sd: {1}'.format(np.mean(syn_densities), np.std(syn_densities)))

    # create Pandas dataframes and write to excel
    axon_syn_densities = pd.DataFrame({'synapse densities': syn_densities})
    excel_writer = pd.ExcelWriter(excel_out_path + 'axon_synapse_densities.xlsx', engine='xlsxwriter')
    axon_syn_densities.to_excel(excel_writer, sheet_name='Axon synapse densities')
    excel_writer.save()

    # take synapse centric view now
    # extract synapse annotation from this file as well
    synapses, _ = sa.synapses_from_jk_anno(bbone_synapses_annos)
    print('Number synapses found: {0}'.format(len(synapses)))

    # check this
    synapses = [s for s in synapses if not 'exclude' in s.tags]

    for syn in synapses:
        # add to each synapse its previously inferred post type
        syn.post_type = syn.postNode.annotation.automatic_classification # one of INT, X, RAP

    syn_sizes = [np.power((s.az_len/1000./2.),2)*np.pi for s in synapses]
    syn_types = [s.post_type for s in synapses]
    syn_location = [s.location_tags[0] for s in synapses]

    RAP_syns = [s for s in synapses if 'RA' == s.post_type]
    X_syns = [s for s in synapses if 'X' == s.post_type]
    INT_syns = [s for s in synapses if 'interneuron' == s.post_type]

    num_tot = float(len(RAP_syns) + len(X_syns) + len(INT_syns))

    # calculate type densities per mm
    mean_out_syn_density = np.mean(syn_densities)
    fraction_RAP = len(RAP_syns) / num_tot
    fraction_X = len(X_syns) / num_tot
    fraction_INT = len(INT_syns) / num_tot

    out_RAP_dens = mean_out_syn_density * fraction_RAP
    out_X_dens = mean_out_syn_density * fraction_X
    out_INT_dens = mean_out_syn_density * fraction_INT

    print('Out RAP density s/mm: {0}'.format(out_RAP_dens))
    print('Out X density s/mm: {0}'.format(out_X_dens))
    print('Out INT density s/mm: {0}'.format(out_INT_dens))

    # calculate est. total number of target post types
    tot_mean_RAP_axon_length = 14.731 # in mm
    print('Num out ht syns per RAP {0}'.format(out_RAP_dens*tot_mean_RAP_axon_length))
    print('Num out X syns per RAP {0}'.format(out_X_dens*tot_mean_RAP_axon_length))
    print('Num out INT syns per RAP {0}'.format(out_INT_dens*tot_mean_RAP_axon_length))

    # create Pandas dataframes and write to excel
    syn_sizes_and_types = pd.DataFrame({'synapse size': syn_sizes,
                                        'post type' : syn_types,
                                        'post location': syn_location})
    excel_writer = pd.ExcelWriter(excel_out_path + 'syn_sizes_and_types.xlsx', engine='xlsxwriter')
    syn_sizes_and_types.to_excel(excel_writer, sheet_name='synapses')
    excel_writer.save()

    return bbone_synapses_annos

def get_suppl_tiff_stacks(out_path='/suppl_example_stacks/',
                          knossos_ds_path='add path to knossos dataset configuration file'):
    """
    Extract subregions from the j0256 dataset and store them as tif stacks.

    :return:
    """

    stack_dims = (100,100,18) # in pixels



    # symmetric synapse stack
    sy_offset = (6240, 7249, 500)

    # asymmetric synapse stack
    as_offset = (6599, 8330, 488)

    sy_out_path = out_path + 'sy_stack/'
    as_out_path = out_path + 'as_stack/'

    try:
        os.mkdir(out_path + 'sy_stack/')
        os.mkdir(out_path + 'as_stack/')
    except:
        print('Could not make substack directories.')

    j0256 = kds.knossosDataset()
    j0256.initialize_from_knossos_path(knossos_ds_path)

    j0256.from_raw_cubes_to_image_stack(stack_dims, sy_offset,
                                        output_path = sy_out_path,
                                        name='supplementary_stack_1',
                                        overwrite=True)

    j0256.from_raw_cubes_to_image_stack(stack_dims, as_offset,
                                        output_path = as_out_path,
                                        name='supplementary_stack_2',
                                        overwrite=True)
    return


def get_synapse_densities_for_axons(axons):
    """
    Calculate synapses densities for the axons, using the annotation scheme from synapseAnalyzer

    :param axons: SkeletonAnnotation objects
    :return:
    """

    syn_densities = []
    for axon in axons:
        # get the number of synapses for this axon
        pre_syn_nodes = [pre_n for pre_n in axon.getNodes() if 'p4' in pre_n.getPureComment() and not 'exclude'
                                                                                           in pre_n.getPureComment()]
        syn_densities.append(len(pre_syn_nodes) / axon.physical_length()*1e6)

    return syn_densities

def prepare_anno_for_bb_spine_counting(dendrite_anno):
    """
    Gets the backbone length and places comments to make sure that backbone endings
    are not accidentally counted as spines. The annotation is modified in place.

    :param dendrite_anno: SkeletonAnnotation
    :return:
    """

    nxg = su.annoToNXGraph(dendrite_anno)[0]
    bbs = [n for n in dendrite_anno.getNodes() if 'bbone' in n.getPureComment()]
    if len(bbs) == 2:
        b1 = bbs[0]
        b2 = bbs[1]
        try:
            dendrite_anno.bb_len = nx.shortest_path_length(nxg, b1, b2, weight='weight') / 1000.
        except:
            print('Error during bbone path length calculation for dendrite {0}'.format(dendrite_anno.comment))
    else:
        # might need different identifier than comment
        raise Exception('Could not identify backbone comments on: {0}'.format(dendrite_anno.comment))

    for neighbor in b1.getNeighbors():
        existing_comment = neighbor.getPureComment()
        neighbor.setPureComment('no_spine' + existing_comment)
    for neighbor in b2.getNeighbors():
        existing_comment = neighbor.getPureComment()
        neighbor.setPureComment('no_spine' + existing_comment)

    dendrite_anno.removeNode(b1)
    dendrite_anno.removeNode(b2)

    return dendrite_anno

def perform_anno_bb_spine_counting(spine_syn_cnt_kdtree, dendrite_anno):
    """
    Perform the automatic spine counting, by using the synapse count data
    supplied in spine_sync_cnt_kdtree. Results are stored as attributes in dendrite_anno

    :param spine_syn_cnt_kdtree: KdTree
    :param dendrite_anno: SkeletonAnnotation
    :return: SkeletonAnnotation
    """
    dendrite_anno.spines = []
    nxg = su.annoToNXGraph(dendrite_anno)[0]
    candidate_spine_nodes = {k for k, v in nxg.degree().iteritems() if v == 1}

    for end_node in candidate_spine_nodes:
        if 'no_spine' in end_node.getPureComment():
            continue

        for curr_node in nx.traversal.dfs_preorder_nodes(nxg, end_node):
            if nxg.degree(curr_node) > 2:
                branch_found = True
                spine_len = nx.shortest_path_length(nxg, end_node, curr_node, weight='weight')
                try:
                    spine_syn_cnt = spine_syn_cnt_kdtree.query_ball_point(end_node.getCoordinate_scaled(), 50.)[0]
                except:
                    spine_syn_cnt = -1

                if spine_len < 1000. or spine_syn_cnt > 1.0:
                    break
                else:
                    end_node.spine_length = spine_len
                    # found a spine; query the syn_cnt_kdtree to get a synapse cnt here!
                    end_node.spine_syn_cnt = spine_syn_cnt
                    dendrite_anno.spines.append(end_node)
                    break

    dendrite_anno.num_spines = len(dendrite_anno.spines)
    dendrite_anno.spine_density = dendrite_anno.num_spines / dendrite_anno.bb_len

    return dendrite_anno

def classify_post_dendrite(spine_syn_cnt_kdtree, dendrite_anno):
    """
    Classify dendrite_anno by using its spine density.

    :param spine_syn_cnt_kdtree: skeleton_utils kdtree
    :param dendrite_anno: SkeletonAnnotation
    :return:
    """

    # measure dendritic backbone length, label end nodes for not counting
    dendrite_anno = prepare_anno_for_bb_spine_counting(dendrite_anno)

    # performs the actual spine counting and sets attributes as result
    dendrite_anno = perform_anno_bb_spine_counting(spine_syn_cnt_kdtree, dendrite_anno)

    if dendrite_anno.spine_density <= 0.11:
        dendrite_anno.automatic_classification = 'interneuron'
    elif dendrite_anno.spine_density > 0.11 and dendrite_anno.spine_density <= 0.46:
        dendrite_anno.automatic_classification = 'RA'
    else:
        dendrite_anno.automatic_classification = 'X'

    return dendrite_anno


def annotations_to_kzip(annos, path='*.k.zip'):
    """
    Save SkeletonAnnotation objects in a kzip knossos container.
    :param annos: iterable of SkeletonAnnotation
    :param path: str
    :return:
    """

    skel_obj = skeleton.Skeleton()
    for a in annos:
        skel_obj.add_annotation(a)

    skel_obj.to_kzip(path)
    return

def post_HVC_RA_axon_analysis(prefix_bbone = '/axon analysis/transsynaptic tracing/from axons/skeletons/',
                              prefix_syn='/axon analysis/transsynaptic tracing/from axons/synapses/',
                              path_ungrouped='*.csv',
                              path_post_grouped = '*.csv',
                              path_failures = '*.csv'):
    """
    Fnction that analyzes the HVC(RA) axons and their outgoing postsynaptic partners.
    This function should be split up into many modular pieces, but grew organically....
    Paths to adjust:
    - Load all axons with all postsynaptic partners
    - Load all synapse annotations from separate annotation files
    - All csv paths for logs

    This function performs:
    - automatic spine counting
    - grouping and spatial matching of post synaptic dendrites to the same cells (if dendrites were traced from different
    synapses)
    - writes csv files to enable further analysis on the data

    :return: iterable of extended skeleton annotation objects (axons),
    iterable of synapse annotation objects (all_synapse)
    """
    # each file contains one axon tree, post nodes, terminal / en passant annotations and all post annotations
    # post annotations can be tagged with exclude
    paths = [prefix_bbone + 'proximal_12_with_all_post_tracings_bbone_annotated.175.k.zip',
             prefix_bbone + 'proximal_14_with_all_post_tracings_bbone_annotated.056.k.zip',
             prefix_bbone + 'proximal_16_with_all_post_tracings_bbone_annotated.067.k.zip',
             prefix_bbone + 'axon_37_6_with_all_post_tracings_bbone_annotated.167.k.zip',
             prefix_bbone + 'axon_14_5_with_all_post_tracings_bbone_annotated.096.k.zip',
             prefix_bbone + 'axon_21_with_all_post_tracings_bbone_annotated.075.k.zip',
             prefix_bbone + 'axon_myelin_1_with_all_post_tracings__bbone_annotated.049.k.zip',
             prefix_bbone + 'axon_myelin_4_with_all_post_tracings_bbone_annotated.158.k.zip',
             prefix_bbone + 'axon_myelin_7_with_all_post_tracings_bbone_annotated.107.k.zip']

    #### NOT ELEGANT: HAS TO BE IN THE SAME ORDER AS PATHS ###
    # note that the synapse annotations for the proximal axons are in the same annotation files as
    # the dendritic tracings (bbone-labeled dendrites)
    paths_synapses = [prefix_bbone + 'proximal_12_with_all_post_tracings_bbone_annotated.158.k.zip',
             prefix_bbone + 'proximal_14_with_all_post_tracings_bbone_annotated.056.k.zip',
             prefix_bbone + 'proximal_16_with_all_post_tracings_bbone_annotated.065.k.zip',
             prefix_syn + 'axon 037,006_with_post_synapses.156.k.zip',
             prefix_syn + 'axon 014,005_with_post_synapses.039.k.zip',
             prefix_syn + 'axon 021_with_post_synapses.032.k.zip',
             prefix_syn + 'myelinated_1_with_post_synapses-150918T1103.074.k.zip',
             prefix_syn + 'myelinated_4_with_post_synapses-150918T2012.176.k.zip',
             prefix_syn + 'myelinated_7_with_post_synapses-150922T1755.099.k.zip']

    # make a kd-tree of all synapse counts at the spine head locations;
    # spines are then later equipped with the counts for further analysis
    _, syn_cnt_kdtree = get_spine_count_data()

    btext = open(path_ungrouped, 'w')
    btext_convergence = open(path_post_grouped, 'w')
    btext_fail_log = open(path_failures, 'w')

    btext_convergence.write('name\tclassification\tpresynapse type\n')
    btext.write('Axon name\tterminal or passant\tpost path length\tnum post spines\tpost spine density [1/um]\tclassification\tavg spine syn cnt\taz len [nm]\tpost location\tpost dark?\tpost file\n')
    axons = []
    all_synapse = []
    group_coords = []
    remaining_spine_cnts = []
    num_excluded_post_dendrites = 0
    excluded_dendrites = []
    boundary_excluded_dendrites = []
    obscured_excluded_dendrites = []
    tracing_excluded_dendrites = []
    wrongly_annotated_axons = []
    HVC_RA_black_dendrite_lengths = []
    HVC_RA_non_black_lengths = []

    glob_num_post_dendrites = 0
    glob_num_syns = 0

    for p, p_syn in zip(paths, paths_synapses):

        annos = su.load_j0256_nml(p)
        annos_synapses = su.load_j0256_nml(p_syn)

        # extract synapses
        axon_synapses, bad = sa.synapses_from_jk_anno(annos_synapses)

        # create synapse spatial lookup
        syn_tree = sa.synapsesToKDtree(axon_synapses, synapse_location='post')
        try:
            axon = [a for a in annos if 'axon' in a.comment][0]
        except:
            print('Failed to extract axon from ' + p)

        axon.outgoing_synapses = []
        axon_tree = su.annoToKDtree(axon)[0]

        # only non-exclude post dendrites / cells
        post_annos = [a for a in annos if not 'axon' in a.comment]
        post_non_exclude = []
        for a in post_annos:
            this_anno_comments = [n.getPureComment() for n in a.getNodes()]
            if True in map(lambda x: 'exclude' in x, this_anno_comments):
                if True in map(lambda x: 'boundary' in x, this_anno_comments):
                    boundary_excluded_dendrites.append(a)
                    excluded_dendrites.append(a)
                    num_excluded_post_dendrites += 1
                elif True in map(lambda x: 'tracing' in x, this_anno_comments):
                    tracing_excluded_dendrites.append(a)
                    excluded_dendrites.append(a)
                    num_excluded_post_dendrites += 1
                elif True in map(lambda x: 'obscured' in x, this_anno_comments):
                    obscured_excluded_dendrites.append(a)
                    excluded_dendrites.append(a)
                    num_excluded_post_dendrites += 1
                elif True in map(lambda x: 'axon' in x, this_anno_comments):
                    wrongly_annotated_axons.append(a)
                #this_axon_excludes.append(a)
                continue
            else:
                post_non_exclude.append(a)

        # spatially match post synaptic dendritic annotations
        matched_post_dendrites, match_graph = su.annotation_matcher(post_non_exclude, skip_if_too_small=False)

        # analyze spine density in all non_exclude
        for p_cell in post_non_exclude:
            glob_num_syns += 1
            p_cell.source_axon = axon.comment
            first_coord = [n.getCoordinate_scaled() for n in p_cell.getNodes() if 'First' in n.getPureComment()]
            first_node_comment = [n.getPureComment() for n in p_cell.getNodes() if 'First' in n.getPureComment()]
            if len(first_node_comment) == 1:
                first_node_comment = first_node_comment[0]
            if first_coord:
                # query axon kd-tree
                found_axon_post_seed = axon_tree.query_k_nearest(first_coord, 1)[0]
                # determine en passant or terminal if annotated
                if 'terminal' in found_axon_post_seed.getPureComment() or 'terminal' in first_node_comment:
                    p_cell.syn_type = 'terminal'
                elif 'passant' in found_axon_post_seed.getPureComment() or 'passant' in first_node_comment:
                    p_cell.syn_type = 'passant'
                else:
                    p_cell.syn_type = 'unclear'
                    btext_fail_log.write('No passant / terminal comment found for ' + p_cell.comment + ' for axon ' + axon.comment + ' add passant or terminal, first coord: ' + '{0}, {1}, {2}; first node comment: {3} + \n'.format(int(first_coord[0][0]/11.), int(first_coord[0][1]/11.), int(first_coord[0][2]/29.), first_node_comment))
            else:
                btext_fail_log.write('Warning, first node comment found for ' + p_cell.comment + ' for axon ' + axon.comment + ' add First node or exclude to dendrite\n')
                p_cell.syn_type = 'unclear'
                first_coord = [[0.,0.,0.,]]
            p_cell.first_coord = first_coord
            btext.write("{0}\t".format(axon.comment))
            btext.write("{0}\t".format(p_cell.syn_type))
            p_cell.spines = []

            # automatic spine classification
            # get all end nodes
            # count those end nodes as spines, that have more than 1 um distance to the next branch point
            nxg = su.annoToNXGraph(p_cell)[0]
            # get backbone length
            b1 = None
            b2 = None
            bbs = [n for n in p_cell.getNodes() if 'bbone' in n.getPureComment()]
            if len(bbs) == 2:
                b1 = bbs[0]
                b2 = bbs[1]
                p_cell.bb_len = nx.shortest_path_length(nxg, b1, b2,\
                                                  weight='weight')/1000.

                # test whether there is an analyze comment between the bbone comments
                path_nodes = nx.shortest_path(nxg, source=b1, target=b2, weight='weight')
                path_comments = [n.getPureComment() for n in path_nodes]
                if not True in map(lambda x: 'analyze' in x, path_comments):
                    btext_fail_log.write(
                        'Warning, no analyze for bbone annotation found for ' + p_cell.comment + ' for axon '\
                        + axon.comment + ' for first coord: ' + '{0}, {1}, {2}'.format(
                            int(p_cell.first_coord[0][0] / 11.), int(p_cell.first_coord[0][1] / 11.),
                            int(p_cell.first_coord[0][2] / 29.)) + '\n')

            else:
                p_cell.bb_len = 0.
                p_cell.automatic_classification = 'NA'
                btext_fail_log.write('Warning, no / broken bbone annotation found for ' + p_cell.comment + ' for axon '\
                                     + axon.comment + ' for first coord: ' +\
                                     '{0}, {1}, {2}'.format(int(p_cell.first_coord[0][0]/11.),
                                                            int(p_cell.first_coord[0][1]/11.),
                                                            int(p_cell.first_coord[0][2]/29.)) + '\n')
                btext.write("\n")
                continue

            p_cell.GT_int = False
            if [n for n in p_cell.getNodes() if 'GT_int' in n.getPureComment()]:
                p_cell.GT_int = True
                # this is a ground truth interneuron dendrite, meaning that it has a soma in the dataset that clearly
                # looks like an interneuron and that the axon makes symmetric synapses;

            # delete bb nodes, perform connected components on networkx graph
            for neighbor in b1.getNeighbors():
                existing_comment = neighbor.getPureComment()
                neighbor.setPureComment('no_spine' + existing_comment)
            for neighbor in b2.getNeighbors():
                existing_comment = neighbor.getPureComment()
                neighbor.setPureComment('no_spine' + existing_comment)
            p_cell.removeNode(b1)
            p_cell.removeNode(b2)
            nxg = su.annoToNXGraph(p_cell)[0]

            ccs = list(nx.connected_component_subgraphs(nxg, copy=False))
            automatic_spine_cnt = 0
            branch_found = False
            for cc in ccs:
                if True in map(lambda x: 'analyze' in x, [n.getPureComment() for n in cc.nodes()]):
                    # this is the correct cc that should be analyzed; this was annotated manually
                    # get end nodes
                    candidate_spine_nodes = {k for k, v in cc.degree().iteritems() if v == 1}

                    # DFS to first branch node
                    for end_node in candidate_spine_nodes:
                        if 'no_spine' in end_node.getPureComment():
                            continue
                        for curr_node in nx.traversal.dfs_preorder_nodes(cc, end_node):
                            if cc.degree(curr_node) > 2:
                                branch_found = True
                                b_len = nx.shortest_path_length(cc, end_node, curr_node, weight='weight')
                                try:
                                    spine_syn_cnt = syn_cnt_kdtree.query_ball_point(end_node.getCoordinate_scaled(), 50.)[0]
                                except:
                                    # this spine has no count annotation so far
                                    spine_syn_cnt = -1

                                if b_len < 1000. or spine_syn_cnt > 1.0:
                                    break
                                else:

                                    if spine_syn_cnt == -1:
                                        # this spine should be counted!
                                        remaining_spine_cnts.append(end_node.getCoordinate())

                                    end_node.spine_length = b_len
                                    # found a spine; query the syn_cnt_kdtree to get a synapse cnt here!
                                    end_node.spine_syn_cnt = spine_syn_cnt
                                    p_cell.spines.append(end_node)
                                    automatic_spine_cnt += 1
                                    break
                    break

            btext.write("{0:3.2f}\t".format(p_cell.bb_len))
            btext.write("{0}\t".format(automatic_spine_cnt))
            p_cell.num_spines = automatic_spine_cnt
            if p_cell.bb_len and p_cell.num_spines:
                btext.write("{0:3.2f}\t".format(p_cell.num_spines/p_cell.bb_len))
            else:
                btext.write("{0}\t".format('NA'))

            # analyze spine synapse count for this p_cell
            spine_syn_cnts = [s.spine_syn_cnt for s in p_cell.spines if not s.spine_syn_cnt == -1]
            p_cell.spine_syn_cnt_NAN = len([s.spine_syn_cnt for s in p_cell.spines if s.spine_syn_cnt == -1])
            if len(spine_syn_cnts) > 0:
                p_cell.spine_syn_cnt_avg = np.mean(spine_syn_cnts)
            else:
                p_cell.spine_syn_cnt_avg = -1.

            p_cell.spine_density = p_cell.num_spines / p_cell.bb_len
            if p_cell.spine_density <= 0.11:
                p_cell.automatic_classification = 'interneuron'
            elif p_cell.spine_density > 0.11 and p_cell.spine_density <= 0.46:
                p_cell.automatic_classification = 'RA'
            else:
                p_cell.automatic_classification = 'X'
            btext.write("{0}\t".format(p_cell.automatic_classification))
            btext.write("{0:3.2f}\t".format(p_cell.spine_syn_cnt_avg))

            if first_coord[0][0] > 0.:
            # attempt to find the corresponding synapse in the synapse tree by spatial matching of the post synapse node
                corresponding_syn, dist = syn_tree.query_k_nearest(first_coord, return_dists=True)
            else:
                btext_fail_log.write('No synapse found for ' + p_cell.comment +\
                                     ' check the synapse annotation. For axon: '\
                                     + axon.comment + ' for first coord: ' + '{0}, {1}, {2}'.format(int(p_cell.first_coord[0][0]/11.),
                                                                                                    int(p_cell.first_coord[0][1]/11.),
                                                                                                    int(p_cell.first_coord[0][2]/29.)) + '\n')
                p_cell.corresponding_syn = 'NA'

            if dist < 1000.:
                # more than 1000 nm cannot be right in this case
                p_cell.corresponding_syn = corresponding_syn[0]
                p_cell.corresponding_syn.post_type =  p_cell.automatic_classification
                p_cell.corresponding_syn.axon = axon
                axon.outgoing_synapses.append(p_cell.corresponding_syn)

            else:
                p_cell.corresponding_syn = 'NA'
                btext_fail_log.write('Synapse way too far: ' + str(dist) + ' nm for ' + p_cell.comment + ' check the synapse annotation. For axon: ' + axon.comment +' for first coord: ' + '{0}, {1}, {2}'.format(int(p_cell.first_coord[0][0]/11.), int(p_cell.first_coord[0][1]/11.), int(p_cell.first_coord[0][2]/29.)) + '\n')

            try:
                p_cell.corresponding_syn.axon = axon
                p_cell.corresponding_syn.dendrite = p_cell
                btext.write("{0:3.2f}\t".format(p_cell.corresponding_syn.az_len))
                btext.write("{0}\t".format(p_cell.corresponding_syn.location_tags[0]))
                if 'bb' in p_cell.corresponding_syn.tags:
                    p_cell.BDA_stained = True
                    btext.write("{0}\t".format('black'))
                else:
                    p_cell.BDA_stained = False
                    btext.write("{0}\t".format('unstained'))
                all_synapse.append(p_cell.corresponding_syn)
            except:
                btext.write("{0:3.2f}\t".format(0.0))
                btext.write("{0}\t".format('NA'))
                btext.write("{0}\t".format('NA'))

            if p_cell.BDA_stained:
                HVC_RA_black_dendrite_lengths.append(p_cell.corresponding_syn.az_len)
            elif p_cell.BDA_stained == False and p_cell.automatic_classification == 'RA':
                HVC_RA_non_black_lengths.append(p_cell.corresponding_syn.az_len)

            btext.write("{0}\t\n".format(p_cell.comment))
            axon.post_dendrites = post_non_exclude

        # analyse match groups - do this now, since only now the post dendrites contain type information
        btext_convergence.write('new axon: {0}\n'.format(axon.filename))
        convergence_analysis = lambda x: x
        convergence_analysis.groups = []

        for group in matched_post_dendrites:
            btext_convergence.write('new group \n')
            post_group = lambda x:x
            post_group.group_spine_density = 0.
            post_group.group_az_len = 0.
            post_group.group_spine_synapse_cnt = 0.
            post_group.az_area = 0.
            post_group.group_dendrites = group
            this_first_coord = (0,0,0)
            cnt = 0
            cnt_syns = 0
            BDA_stained = False
            GT_int = False
            for post_dendrite in group:
                glob_num_post_dendrites += 1
                try:
                    post_group.group_spine_density += post_dendrite.spine_density
                    cnt += 1
                except:
                    btext_fail_log.write('Warning, dendrite ' + post_dendrite.comment +\
                                         ' has no spine density, check bbone and the analyze comment. For axon: '\
                                         + axon.comment +' for first coord: ' + '{0}, {1}, {2}'.format(int(post_dendrite.first_coord[0][0]/11.),
                                                                                                       int(post_dendrite.first_coord[0][1]/11.),
                                                                                                       int(post_dendrite.first_coord[0][2]/29.)) + '\n')
                    post_dendrite.spine_density = -1.
                if post_dendrite.first_coord[0][0] > 0.:
                    this_first_coord = (int(post_dendrite.first_coord[0][0]/11.),
                                        int(post_dendrite.first_coord[0][1]/11.),
                                        int(post_dendrite.first_coord[0][2]/29.))

                try:
                    post_dendrite.az_len = post_dendrite.corresponding_syn.az_len
                    post_dendrite.syn_post_location = post_dendrite.corresponding_syn.location_tags[0]
                    post_group.group_az_len += post_dendrite.az_len
                    post_group.az_area += (((post_dendrite.az_len/2.)**2)*np.pi)

                    cnt_syns += 1
                    post_dendrite.corresponding_syn.conv_index = cnt_syns
                except:
                    # this means that no synapse could be matched

                    btext_fail_log.write('Warning, dendrite ' + post_dendrite.comment +\
                                         ' has no synapse attached, add / fix synapse at First node. For axon: '\
                                         + axon.comment +' for first coord: ' + '{0}, {1}, {2}'.format(int(post_dendrite.first_coord[0][0]/11.),
                                                                                                       int(post_dendrite.first_coord[0][1]/11.),
                                                                                                       int(post_dendrite.first_coord[0][2]/29.)) + '\n')
                    post_dendrite.az_len = 0.
                    post_dendrite.syn_post_location = 'NA'
                if post_dendrite.GT_int:
                    GT_int = True

                try:
                    if post_dendrite.BDA_stained:
                        # it is sufficient that one dendrite tracing was found to be stained to say the biological dendrite
                        # is black
                        BDA_stained = True
                except:
                    BDA_stained = False

                btext_convergence.write("{0}\t".format(post_dendrite.comment))
                btext_convergence.write("{0}\t".format(post_dendrite.automatic_classification))
                btext_convergence.write("{0:3.2f}\t".format(post_dendrite.spine_density))
                btext_convergence.write("{0:3.2f}\t".format(post_dendrite.az_len))
                btext_convergence.write("{0}\t".format(post_dendrite.syn_post_location))
                btext_convergence.write("{0}\t\n".format(post_dendrite.syn_type))
            if cnt_syns > 0:
                post_group.group_az_len /= cnt_syns
            else:
                post_group.group_az_len = 0.
            btext_convergence.write('avg group az len: {0:3.2f}\t'.format(post_group.group_az_len))
            if cnt > 0:
                post_group.group_spine_density /= cnt
            else:
                post_group.group_spine_density = - 1.

            if GT_int:
                post_group.GT_int = True
                btext_convergence.write('GT interneuron\t')
            else:
                post_group.GT_int = False
                btext_convergence.write('no GT interneuron\t')

            if BDA_stained:
                post_group.BDA_stained = True
                btext_convergence.write('black\t')
            else:
                post_group.BDA_stained = False
                btext_convergence.write('unstained\t')

            btext_convergence.write('avg group spine density: {0:3.2f}\t'.format(post_group.group_spine_density))
            if post_group.group_spine_density <= 0.11:
                btext_convergence.write('{0}\t\n'.format('interneuron'))
                post_group.group_classification = 'interneuron'
            elif post_group.group_spine_density > 0.11 and post_group.group_spine_density <= 0.46:
                btext_convergence.write('{0}\t\n'.format('RA'))
                post_group.group_classification = 'RA'
            else:
                btext_convergence.write('{0}\t\n'.format('X'))
                post_group.group_classification = 'X'

            # correct the individual axon synapses with the new type
            # classification from the post dendrites grouped together
            for post_dendrite in post_group.group_dendrites:
                if not post_dendrite.corresponding_syn == 'NA':
                    try:
                        post_dendrite.corresponding_syn.post_type =  post_group.group_classification
                    except:
                        post_dendrite.corresponding_syn.post_type = 'NA'

            if post_group.group_spine_density < 0.8 and '37' in axon.comment:
                group_coords.append(this_first_coord)

            convergence_analysis.groups.append(post_group)
        convergence_analysis.match_graph = match_graph
        axon.convergence_analysis = convergence_analysis
        axons.append(axon)

    btext.close()
    btext_convergence.close()
    btext_fail_log.close()

    all_spines = []
    for a in axons:
        for p in a.post_dendrites:
            all_spines.extend(p.spines)

    print('Number of total spines found: ' + str(len(all_spines)))
    print('Number of excluded dendrites: ' + str(num_excluded_post_dendrites))
    print('Number of obscured exclude dendrites: ' + str(len(obscured_excluded_dendrites)))
    print('Number of boundary exclude dendrites: ' + str(len(boundary_excluded_dendrites)))
    print('Number of tracing exclude dendrites: ' + str(len(tracing_excluded_dendrites)))
    print('Number of wrongly annotated axons that are not further considered: ' + str(len(wrongly_annotated_axons)))

    print('Global number post dendrites: ' +str(glob_num_post_dendrites))
    print('Global number syns: ' + str(glob_num_syns))

    return axons, all_synapse


def add_main_branch_end_nodes_to_axons(axons, path_to_annotations = '/axon analysis/transsynaptic tracing/from axons/main branch annotations/'):
    """
    Adds manually created main branch annotation markers to the skeleton annotation objects for further analysis.

    :param axons: iterable of SkeletonAnnotation objects
    :param path_to_annotations: str
    :return:
    """
    paths = [path_to_annotations + 'myelinated_1_with_post_synapses-_with_main_branch.057.k.zip',
             path_to_annotations + 'myelinated_4_with_post_synapses-_with_main_branch.125.k.zip',
             path_to_annotations + 'myelinated_7_with_post_synapses_with_main_branch.055.k.zip',
             path_to_annotations + 'proximal_axon_12_with_main_branch.001.k.zip',
             path_to_annotations + 'proximal_axon_14_with_main_branch.002.k.zip',
             path_to_annotations + 'proximal_axon_16_with_main_branch.001.k.zip',
             path_to_annotations + 'axon 014,005_with_main_branch.030.k.zip',
             path_to_annotations + 'axon 021_with_main_branch.017.k.zip',
             path_to_annotations + 'axon 037,006_with_main_branch.162.k.zip']

    axons_with_main_branchs = [su.load_j0256_nml(p)[0] for p in paths]
    for a_m in axons_with_main_branchs:
        # find all main branch end coords for this axon
        end_coords = [n.getCoordinate_scaled() for n in a_m.getNodes() if 'main_branch' in n.getPureComment()]
        # match the current axon to the right axon in the parameter axons
        # do an ugly O(n^2) matching procedure
        matched = False
        for a in axons:
            if end_coords[0] in [n.getCoordinate_scaled() for n in a.getNodes()]:
                a.main_branch_end_markers = end_coords
                matched = True
                break
        if not matched:
            raise Exception('Could not match ' + a_m.filename + ' to any axon.')

    return axons


def add_main_branch_densities_to_axons(axons):
    """
    Adds the manually annotated main branch densities to the Skeleton annotations of the axons.

    :param axons: iterable of SkeletonAnnotation objects
    :return:
    """
    # branch density required for each axon, add it
    y_main_branch_densites_and_names = [(3.498, 'axon_myelin_1'),
                                      (11.446, 'axon_myelin_4'),
                                      (6.279, 'axon_myelin_7'),
                                      (10.579, 'axon_12'),
                                      (16.681, 'axon_14'),
                                      (9.930, 'axon_16'),
                                      (2.796, 'axon_14_5'),
                                      (0.000, 'axon_21'),
                                      (0.000, 'axon_37_6')]

    ax_bd_lookup = dict([(a[1], a[0]) for a in y_main_branch_densites_and_names])
    ax_soma_true = dict([('axon_12', True),('axon_14', True), ('axon_16', True)])

    y_sorted = sorted(y_main_branch_densites_and_names, key=lambda x: x[0])
    #print y_sorted

    # axons are sorted by branch density AND existence of soma inside dataset, as this is more informative about the
    # actual soma distance of the axonal part compared to the branch density metric
    y_sorted[-2], y_sorted[-4] = y_sorted[-4], y_sorted[-2]
    y_sorted[-2], y_sorted[-3] = y_sorted[-3], y_sorted[-2]
    #print y_sorted
    # get axons into the order above
    re_ordered_axons = []
    for a_sorted in y_sorted:
        re_ordered_axons.append([a for a in axons if a_sorted[1] in a.comment][0])

    for a in re_ordered_axons:
        a.branch_density = ax_bd_lookup[a.comment]
        if ax_soma_true.has_key(a.comment):
            a.soma_true = True

    return re_ordered_axons


def make_orphan_labeling_estimate_tasks(path_to_out_folder='...'):
    def add_and_link_node(anno, new_node_coordx, new_node_coordy, new_node_coordz, comment='', src_node=None):
        trg_node = skeleton.SkeletonNode()
        trg_node.from_scratch(anno, new_node_coordx, new_node_coordy, new_node_coordz)
        trg_node.setPureComment(comment)
        anno.addNode(trg_node)
        if src_node:
            src_node.addChild(trg_node)

        return trg_node

    num_samples = 1000
    samples_per_file = 50
    scaling = (11., 11., 29.)

    b_size = np.array([91., 91., 35.])

    # generate random boxes
    dataset_offset = np.array((0, 0, 0))
    dataset_boundaries = np.array((15105, 15105, 2662))
    border_margins = np.array((91, 91, 35))
    sampling_region = zip(dataset_offset + border_margins, dataset_boundaries - border_margins)

    sample_offsets = [map(lambda x: randint(*x), sampling_region) for n in range(num_samples)]
    sample_chunks = chunks(sample_offsets, samples_per_file)

    file_cnt = 0
    sample_cnt = 0
    for chunk in sample_chunks:
        skel_obj = skeleton.Skeleton()
        new_anno = skeleton.SkeletonAnnotation()
        new_anno.scaling = scaling
        skel_obj.add_annotation(new_anno)
        # skel_obj.scaling = scaling

        for s_off in chunk:
            # create a linked box of nodes
            n1 = add_and_link_node(new_anno, *s_off, comment='ID: ' + str(sample_cnt) + ' offset ' + str(s_off))
            n2 = add_and_link_node(new_anno, s_off[0] + b_size[0], s_off[1], s_off[2], comment='todo', src_node=n1)
            n3 = add_and_link_node(new_anno, s_off[0] + b_size[0], s_off[1] + b_size[1], s_off[2], src_node=n2)
            n4 = add_and_link_node(new_anno, s_off[0], s_off[1] + b_size[1], s_off[2], src_node=n3)
            n4.addChild(n1)

            for z_layer in range(int(s_off[2]), int(s_off[2] + b_size[2]), 2):
                n5 = add_and_link_node(new_anno, s_off[0], s_off[1], z_layer, src_node=n1)
                n6 = add_and_link_node(new_anno, s_off[0] + b_size[0], s_off[1], z_layer, src_node=n5)
                n7 = add_and_link_node(new_anno, s_off[0] + b_size[0], s_off[1] + b_size[1], z_layer, src_node=n6)
                n8 = add_and_link_node(new_anno, s_off[0], s_off[1] + b_size[1], z_layer, src_node=n7)
                n8.addChild(n5)

                n1.addChild(n5)
                n2.addChild(n6)
                n3.addChild(n7)
                n4.addChild(n8)

            sample_cnt += 1
        outfile = path_to_out_folder + 'orphan_est_' + str(file_cnt) + '.k.zip'

        skel_obj.to_kzip(outfile, force_overwrite=True)
        print('Writing ' + outfile)

        file_cnt += 1

    return


def read_orphan_labeling_estimate_tasks(path_to_folder='/axon analysis/orphan labeling efficiency/'):
    """
    Read the results of the orphan tracing tasks and calculate the resulting orphan labeling efficiency.

    :param path_to_folder:
    :return:
    """

    # Calculate estimate of expected HVC(RA) path length per sample cube
    # Total path length of HVC_RA inside HVC, based on measurements by S. Benezra and #40k HVC(RA): 589240.in mm
    # Vol estimate of HVC 0.35 mm^3, based on measurements by S. Benezra
    exp_axon_density = 589240. * 1e3 / (0.35 * 1e9)

    files = glob.glob(path_to_folder + '*.k.zip')

    n_box_analyzed = 0
    total_orphan_path_len = 0.
    num_orphans_found = 0
    box_size = np.array([91., 91., 35.])
    len_per_cube = defaultdict(float)

    results_for_knossos_inspection = skeleton.Skeleton()

    for f in files:
        annos = su.load_j0256_nml(f)

        # get all box coordinates
        box_anno = [a for a in annos if a.physical_length() > 10000.][0]
        box_coords = [np.array(n.getCoordinate_scaled())
                      for n in box_anno.getNodes() if 'ID:' in n.getPureComment()]

        results_for_knossos_inspection.add_annotation(box_anno)

        if len(box_coords) != 50:
            raise Exception('Number of sample box coords wrong: ' + f)

        # find closest box
        box_tree = su.KDtree(box_coords, coords=box_coords)  # use the box coords themselves as return objects

        # find the number of boxes in neuropil - boxes that were randomly placed outside in other regions are discared
        # n_neuropil = len([n for n in box_anno.getNodes() if 'done' in n.getPureComment()])
        n_box_analyzed += len(box_coords)

        orphan_annos = [a for a in annos if a.physical_length() < 10000.]

        # map orphans to (neuropil) boxes
        # calculate center of mass of orphan tracing
        for o_a in orphan_annos:
            com_coords = np.array([n.getCoordinate_scaled() for n in o_a.getNodes()])
            com = (com_coords[:, 0].mean(), com_coords[:, 1].mean(), com_coords[:, 2].mean())
            box_coord, dist = box_tree.query_k_nearest(com, k=1, return_dists=True)

            # Interpolate nodes on orphan tracings. This is important, since outside nodes will be pruned and the length
            # will be measured afterwards.
            o_a.interpolate_nodes(max_node_dist_scaled=40)
            o_a_nodes = o_a.getNodes()
            num_orphans_found += 1

            box_size_2 = np.array([91. * 11., 91. * 11., 35. * 29.])

            # Get nodes that are outside bounding box
            outside_box = [n for n in o_a_nodes if
                           (n.getCoordinate_scaled()[0] < box_coord[0] or \
                            n.getCoordinate_scaled()[0] > (box_coord[0] + box_size_2[0])) \
                           or \
                           (n.getCoordinate_scaled()[1] < box_coord[1] or \
                            n.getCoordinate_scaled()[1] > (box_coord[1] + box_size_2[1])) \
                           or \
                           n.getCoordinate_scaled()[2] < box_coord[2] or \
                           n.getCoordinate_scaled()[2] > (box_coord[2] + box_size_2[2])]

            for node in outside_box:
                o_a.removeNode(node)

            results_for_knossos_inspection.add_annotation(o_a)
            len_per_cube[str(box_coord)] += (o_a.physical_length() / 1000.)

            print('Orphan length: ' + str(o_a.physical_length() / 1000.))

            # Add orphan path length
            total_orphan_path_len += o_a.physical_length()

    # The est_orphan_density is calculated as: total_orphan_path_len / n_box_analyzed * box_vol
    print('Total orphan length: ' + str(total_orphan_path_len))
    est_orphan_density = total_orphan_path_len / 1e3 / (
    n_box_analyzed * np.product(np.multiply(box_size, (11., 11., 29.))) / 1e9)
    print('# samples anaylized: ' + str(n_box_analyzed))
    print('# orphans found: ' + str(num_orphans_found))
    print('Expected path density [um/um^3]: ' + str(exp_axon_density))
    print('Found orphan path density [um/um^3]: ' + str(est_orphan_density))
    print('Resulting orphan labeling efficiency: ' + str(est_orphan_density / exp_axon_density))

    try:
        os.makedirs(path_to_folder + 'pruned_results/')
    except OSError:
        if not os.path.isdir(path_to_folder + 'pruned_results/'):
            raise

    results_for_knossos_inspection.to_kzip(path_to_folder + 'pruned_results/pruned_all.k.zip', force_overwrite=True)

    return


#######################################################################################################################
## 3D visualization functions follow
#######################################################################################################################

def plot_HVC_RA_cells_with_direct_syns(cells, syns_same_size=True):
    """
    Visualize found direct synapses on all 12 HVC(RA) cells analyzed (Fig.1 ).
    Use the function load_direct_HVC_RA_HVC_RA_synapses() to get the data.

    :param cells: iterable of annotation objects
    :param syns_same_size: bool
    :return:
    """

    for c in cells:
        print c.filename
        c.color = (0.,0.,0.)
        plt.visualize_annotation(c, dataset_identifier='j0256', show_outline=False)
        # plot scale sphere
        skelplt.add_spheres_to_mayavi_window((0.,0.,0.,), [20000.], color=(0.,0.,0.,1.))
        # plot soma
        plt.add_spheres_to_mayavi_window([c.soma_node.getCoordinate_scaled()],
                                         radii=[8000.], color=(0.,0.,0.,1.))
        if syns_same_size:
            s_coords = [s.preNode.getCoordinate_scaled() for s in c.confirmed_syns]
            plt.add_spheres_to_mayavi_window(s_coords, [4000.]*len(s_coords), color=(1.,0.,0.,1.))
        else:
            plt.add_synapses_to_mayavi_window(c.confirmed_syns, color=(1., 0.,0., 1.),
                                              diameter_scale=15.)
        mlab.title(c.filename)

def plot_HVC_RA_001_with_syns(HVC_RA_001, HVC_syns):
    """
    Visualize the synaptome of Fig. 1 with Mayavi. Use load_HVC_RA_001_annotation() to get the data.
    Be aware that "radii" actually stands for the diameter, mayavi is confusing there.

    :param HVC_RA_001: skeleton annotation object
    :param HVC_syns:  iterable of synapse annotation objects
    :return:
    """

    # black skeleton: always adjust node radius and edge radius together for good looking skeletons
    HVC_RA_001.color=(0.,0.,0.)
    plt.visualize_annotation(HVC_RA_001,
                             override_node_radius = 200.,
                             edge_radius = 100.0,
                             dataset_identifier='j0256', show_outline=False)

    # sort synapses for different colors
    sy_s = [s for s in HVC_syns if 'sy' in s.type_tags]
    as_s = [s for s in HVC_syns if 'as' in s.type_tags]
    direct_RA_RA_s = [s for s in HVC_syns if 'bb' in s.tags]

    # visualize soma
    plt.add_spheres_to_mayavi_window([4709*11., 11050*11., 1795*29.],
                                     radii=[8000.], color=(0.,0.,0.,1.))

    #plt.add_synapses_to_mayavi_window(sy_s, color=(0.2, 0.2,1., 1.),
    #                                  diameter_scale=4., all_same_size = 0.)
    #plt.add_synapses_to_mayavi_window(as_s, color=(0.6, 0.6,0.6, 1.),
    #                                  diameter_scale=4., all_same_size = 0.)
    # visualize direct synapses
    #plt.add_synapses_to_mayavi_window(direct_RA_RA_s, color=(0., 0,1., 1.),
    #                                  diameter_scale=4)

    return

def plot_axons_with_synapses(axons):
    """
    Visualize axons with postsynaptic identity.

    :param axons: iterable of skeleton annotation objects
    :return:
    """

    for a in axons:
        a.color = (0.,0.,0.)
        skelplt.visualize_annotation(a,
                                     override_node_radius = 700.,
                                     edge_radius = 350.0,
                                     dataset_identifier='j0256',
                                     show_outline=False)

        # separate the three classes, to allow plotting in different colors
        int_syns = [s for s in a.outgoing_synapses if 'interneuron' == s.post_type]
        RA_syns = [s for s in a.outgoing_synapses if 'RA' == s.post_type]
        X_syns = [s for s in a.outgoing_synapses if 'X' == s.post_type]

        # add soma spheres for the skeletons that have a soma in the dataset
        mlab.title(a.comment)
        if 'axon_12' in a.comment:
            skelplt.add_spheres_to_mayavi_window((8788 * 11., 7660 * 11., 1615*29.), [15000.], color=(0.,0.,0.,1.))
        elif 'axon_14' == a.comment:
            skelplt.add_spheres_to_mayavi_window((8995 * 11., 9852 * 11., 1413*29.), [15000.], color=(0.,0.,0.,1.))
        elif 'axon_16' in a.comment:
            skelplt.add_spheres_to_mayavi_window((9771 * 11., 12237 * 11., 1585*29.), [15000.], color=(0.,0.,0.,1.))

        # this can be used to visualize also the postsynaptic dendrite.
        #if 'axon_14_5' in a.comment:
            # plot additionally one of the post synaptic RAP for illustration
            #skelplt.add_anno_to_mayavi_window(a.post_rap_example, dataset_identifier='j0256')

        # scale sphere
        skelplt.add_spheres_to_mayavi_window((0.,0.,0.,), [20000.], color=(0.,0.,0.,1.))

        # add main branch end node markers
        skelplt.add_spheres_to_mayavi_window(a.main_branch_end_markers, [2000.]*len(a.main_branch_end_markers), color=(0.5,0.5,0.5,1.))

        skelplt.add_synapses_to_mayavi_window(int_syns, synapse_location = 'pre', color=(0.0, 0.43, 0.75, 1.0), diameter_scale=1., all_same_size = 5000.)
        skelplt.add_synapses_to_mayavi_window(RA_syns, synapse_location = 'pre', color=(1.0, 0.0, 0.0, 1.0), diameter_scale=1., all_same_size = 5000.)
        skelplt.add_synapses_to_mayavi_window(X_syns, synapse_location = 'pre', color=(0.0, 0.69, 0.3, 1.0), diameter_scale=1., all_same_size = 5000.)

        # set resolution for png saving
        mlab.gcf().scene.z_plus_view()
        #plot_path =  '...' + a.comment + '_z-view_2_um_spheres' + '.vrml'
        #print(plot_path)
        #mlab.savefig(plot_path, magnification=3)
        #mlab.close()
        #break
        #mlab.gcf().close()

    return