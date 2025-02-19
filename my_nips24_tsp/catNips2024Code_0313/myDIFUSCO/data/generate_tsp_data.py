import argparse,os,sys
import pprint as pp
import time
import warnings
from multiprocessing import Pool
sys.path.append("./")
import lkh
import numpy as np
import tqdm
import tsplib95
#from concorde.tsp import TSPSolver  # https://github.com/jvkersch/pyconcorde
from scipy.spatial.transform import Rotation as R

warnings.filterwarnings("ignore")



if __name__ == "__main__":
  cfd = os.path.dirname(os.path.abspath(__file__))
  sys.path.append(cfd)
  print("cfd = ",cfd)
  parser = argparse.ArgumentParser()
  parser.add_argument("--min_nodes", type=int, default=50)
  parser.add_argument("--max_nodes", type=int, default=50)
  parser.add_argument("--num_samples", type=int, default=128)
  parser.add_argument("--batch_size", type=int, default=16)
  parser.add_argument("--filename", type=str, default=cfd+'/cat.txt')
  parser.add_argument("--solver", type=str, default="lkh")
  parser.add_argument("--lkh_trails", type=int, default=1000)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--my_sample_type",type=str,default='Uniform',help='Uniform|Guass')
  parser.add_argument("--Guass_mean", type=float, default=0)
  parser.add_argument("--Guass_std", type=float, default=1)
  parser.add_argument("--my_point_dim", type=int, default=3)
  parser.add_argument("--my_dim_mode", type=str, default="ddim",help="'ddim'|'d'")
#  parser.add_argument("--my_data_mode", type=str, default="plus",help="'plus'|''")

  opts = parser.parse_args()


  assert opts.num_samples % opts.batch_size == 0, "Number of samples must be divisible by batch size"
  assert opts.my_dim_mode=="ddim" or opts.my_dim_mode=="d", "undefined my_dim_mode!!!"
  np.random.seed(opts.seed)

  if opts.filename is None:
    opts.filename = f"tsp{opts.min_nodes}-{opts.max_nodes}_concorde.txt"

  # Pretty print the run args
  pp.pprint(vars(opts))

  with open(opts.filename, "w") as f:
    start_time = time.time()
    for b_idx in tqdm.tqdm(range(opts.num_samples // opts.batch_size)):
      num_nodes = np.random.randint(low=opts.min_nodes, high=opts.max_nodes + 1)
      assert opts.min_nodes <= num_nodes <= opts.max_nodes
      if opts.my_sample_type == 'Uniform':
        #batch_nodes_coord = np.random.random([opts.batch_size, num_nodes, 2])
        batch_nodes_coord = np.random.random([opts.batch_size, num_nodes, opts.my_point_dim]) *10
      elif opts.my_sample_type == 'Guass':
        #Guass_mean = np.ones(2)*opts.Guass_mean
        #Guass_std = np.eye(2)*opts.Guass_std
        #tmp_data = np.random.normal(opts.Guass_mean, opts.Guass_std, opts.batch_size*num_nodes*2)
        tmp_data = np.random.normal(opts.Guass_mean, opts.Guass_std, opts.batch_size*num_nodes*opts.my_point_dim)
        tmp_data = np.abs(tmp_data) % 10
        #batch_nodes_coord = tmp_data.reshape((opts.batch_size, num_nodes, 2))
        batch_nodes_coord = tmp_data.reshape((opts.batch_size, num_nodes, opts.my_point_dim))
      else:
        assert 0,"undefined sample type"
      for i in range(batch_nodes_coord.shape[0]):
        batch_nodes_coord[i] = batch_nodes_coord[i] - np.mean(batch_nodes_coord[i],axis=0)
      #batch_nodes_coord = batch_nodes_coord -  np.mean(batch_nodes_coord,axis=1) #归一化为0-1范围
      
      if opts.my_dim_mode == "ddim":
        print("opts.my_dim_mode = ",opts.my_dim_mode)
        for i in range(batch_nodes_coord.shape[0]):
          batch_nodes_coord[i,:,2] = 0
          rotation_matrix =  R.random().as_matrix()
          batch_nodes_coord[i] = batch_nodes_coord[i].dot(rotation_matrix)
        #batch_nodes_coord[:,2] = 0
        #rotation_matrix =  R.random().as_matrix()
        #batch_nodes_coord = batch_nodes_coord.dot(rotation_matrix)
#    if opts.my_data_mode == "plus":
#      pass
      
      
      print("")
      
      
      def solve_tsp(parms):
        nodes_coord,point_dim = parms[0],parms[1]
        if opts.solver == "concorde":
          pass
          #scale = 1e6
          #solver = TSPSolver.from_data(nodes_coord[:, 0] * scale, nodes_coord[:, 1] * scale, norm="EUC_2D")
          #solution = solver.solve(verbose=False)
          #tour = solution.tour
        elif opts.solver == "lkh":
          scale = 1e6
          ## my
          ## lkh_path = 'LKH-3.0.6/LKH'
          #lkh_path = '/remote-home/share/worm/wormICML2024Code/LKH-3.0.6/LKH'
          lkh_path = 'catNips2024Code_0313/LKH-3.0.6/LKH'
          problem = tsplib95.models.StandardProblem()
          problem.name = 'TSP'
          problem.type = 'TSP'
          problem.dimension = num_nodes
          if point_dim==2:
            problem.edge_weight_type = 'EUC_2D'
          elif point_dim==3:
            problem.edge_weight_type = 'EUC_3D'
          else:
            assert 0, "Undefined point_dim!!!"
          problem.node_coords = {n + 1: nodes_coord[n] * scale for n in range(num_nodes)}

          solution = lkh.solve(lkh_path, problem=problem, max_trials=opts.lkh_trails, runs=10)
          tour = [n - 1 for n in solution[0]]
        else:
          raise ValueError(f"Unknown solver: {opts.solver}")
        return tour


      with Pool(opts.batch_size) as p:
        tours = p.map(solve_tsp, [[batch_nodes_coord[idx],opts.my_point_dim] for idx in range(opts.batch_size)])

      for idx, tour in enumerate(tours):
        if (np.sort(tour) == np.arange(num_nodes)).all():
          if opts.my_point_dim==2:
            f.write(" ".join(str(x) + str(" ") + str(y) for x, y in batch_nodes_coord[idx]))
          elif opts.my_point_dim==3:
            f.write(" ".join(str(x) + str(" ") + str(y) + str(" ") + str(z) for x, y, z in batch_nodes_coord[idx]))
          else:
             assert 0, "Undefined point_dim!!!"
          
          f.write(str(" ") + str('output') + str(" "))
          f.write(str(" ").join(str(node_idx + 1) for node_idx in tour))
          f.write(str(" ") + str(tour[0] + 1) + str(" "))
          f.write("\n")

    end_time = time.time() - start_time

    assert b_idx == opts.num_samples // opts.batch_size - 1

  print(f"Completed generation of {opts.num_samples} samples of TSP{opts.min_nodes}-{opts.max_nodes}.")
  print(f"Total time: {end_time / 60:.1f}m")
  print(f"Average time: {end_time / opts.num_samples:.1f}s")
