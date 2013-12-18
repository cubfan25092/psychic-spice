#!/usr/bin/env python2

import numpy as np;
import sys;
import os;
import matplotlib.pyplot as pl;

def plot(ws, bs, lams,vs):
	print ws;
	pl.figure(0);
	x = np.linspace(1,1000,1000);
	results = np.zeros((1000));
	for j in xrange(1):
		for i in xrange(1000):
			results[i] = np.linalg.norm(ws[i+(j*1000)]);
		pl.plot(x,results, label=str(lams[j]));
	pl.legend();		
	pl.show();
	figure(1);
	error = np.zeros((1000));
	
	return 0;

def run_epoch_pt1(epoch_num, lam, xy):
	steplength = 1.0/((.01 * epoch_num) + 50);
	w = np.zeros((1,6));
	b = 0;
	for i in xy:
		yk = i[6];
		xk = i[0:6];
		if np.dot(w, xk)+ b >= 1:
			w = w - (steplength * (lam * w));
		else:
			w = w - (steplength * ((lam * w) - (yk * xk)));
			b = b - (steplength *((-1) * yk));
	return w, b;


def pt1(xy):
	for x in xrange(6):
		col = xy[:, x];
		col = (col - np.mean(col))/np.std(col);
		xy[:,x] = col;
	validation_set = xy[:1000, :];
	test_set = xy[1001:6001, :];
	rest = xy[6002:, :];
	w = np.zeros((10*1000,6));
	b = np.zeros((10*1000));
	lam = 1e-7;
	curr_best = 0;
	curr_best_lambda = 0;
	total_epochs = 0;
	results = np.zeros((10));
	for j in xrange(1):
		results[j] = lam;
		for k in xrange(1000):
			np.random.shuffle(rest);
			w[total_epochs], b[total_epochs] = run_epoch_pt1(k, lam, rest[:1000, :]);
			total_epochs += 1;
		lam *= 10;
		print "new lambda generated"
	plot(w,b,results,validation_set);
	


def parse(fname):
	if not os.path.exists(fname):
		return -1;
	raw_data = np.loadtxt(fname, dtype = str, delimiter = ", " );
	includes_pt1 = [0,2,4,10,11,12,14];
	numeric_data = raw_data[:,includes_pt1];
	for x in numeric_data:
		if x[6] == "<=50K":
			x[6] = "-1"
		else:
			x[6] = "1"
	return pt1(numeric_data.astype(np.float));


def main(argv = None):
	if argv is None:
		argv = sys.argv;
	if len(sys.argv) < 2:
		return parse("adult.data");
	else:
		return parse(sys.argv[1]);




if __name__ == "__main__":
	sys.exit(main());