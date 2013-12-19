#!/usr/bin/env python2

import numpy as np;
import sys;
import os;
import matplotlib.pyplot as pl;

def plot(lambdas, acc, wtw):
	pl.figure(0);
	x = np.linspace(1,1000,1000);
	for j in xrange(10):
		pl.plot(x,wtw[1000*j:1000*(j+1)], label=str(lambdas[j]));
	pl.legend();		
	pl.show();
	pl.figure(1);
	for j in xrange(10):
		pl.plot(x,acc[1000*j:1000*(j+1)], label=str(lambdas[j]));
	pl.legend();
	pl.show();
	return 0;

def run_epoch_pt1(epoch_num, lam, xy, vs):
	steplength = 1.0/((.01 * epoch_num) + 50);
	w = np.zeros((1,6));
	b = 0;
	num_correct = 0.0;
	for i in xy:
		yk = i[6];
		xk = i[0:6];
		if np.dot(w, xk)+ b >= 1:
			w = w - (steplength * (lam * w));
		else:
			w = w - (steplength * ((lam * w) - (yk * xk)));
			b = b - (steplength *((-1) * yk));
	for i in vs:
		yk = i[6];
		xk = i[0:6];
		d = np.dot(w, xk);
		if yk * (d + b) >= 1:
			num_correct += 1.0;
	return w, b, num_correct/1000.0;


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
	total_epochs = 0;
	lambdas = np.zeros((10));
	wtw = np.zeros((10*1000));
	acc = np.zeros((10*1000));
	for j in xrange(10):
		lambdas[j] = lam;
		for k in xrange(1000):
			w[total_epochs], b[total_epochs], acc[total_epochs] = run_epoch_pt1(k+1, lam, rest[np.random.choice(rest.shape[0], 1000),:], validation_set);
			wtw[total_epochs] = np.linalg.norm(w[total_epochs]);
			total_epochs += 1;
		lam *= 10;
	plot(lambdas, acc, wtw);
	


def parse(fname):
	if not os.path.exists(fname):
		return -1;
	raw_data = np.loadtxt(fname, dtype = str, delimiter = ", " );
	includes_pt1 = [0,2,4,10,11,12,14];
	numeric_data = raw_data[:,includes_pt1];
	for x in numeric_data:
		if x[6] == "<=50K":
			x[6] = "-1";
		else:
			x[6] = "1";
	np.random.shuffle(numeric_data);
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