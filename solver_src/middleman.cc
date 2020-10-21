#include "datastructure.h"
#include "util.h"
#include "io.h"
#include "viterbi.h"

#ifdef CPLEX_HW
#include "cplex4-hw.h"
#elif CPLEX_HB
#include "cplex4-hb.h"
#elif CPLEX5
#include "cplex5.h"
#elif CPLEX_PZ
#include "cplex5-pz.h"
#else
#include "cplex4.h"
#endif

#include <chrono>
#include <map>
#include <utility>
#include <memory>
#include <stdio.h>
#include <string>
#include <string.h>

const std::string kUsage =
    "./middleman "
    "--per_core_cost=<per_core_cost>\n\t--per_bit_transit_cost=<per_bit_transit"
    "_cost>\n\t--topology_file=<topology_file>\n\t"
    "--middlebox_spec_file=<middlebox_spec_file>\n\t--traffic_r"
    "equest_file=<traffic_request_file>\n\t--algorithm=<algorithm>";

std::vector<middlebox> middleboxes;
std::vector<traffic_request> traffic_requests;
std::vector<node> nodes;
std::vector<std::vector<edge_endpoint>> graph;
std::vector<double> closeness;
std::vector<std::vector<middlebox_instance>> deployed_mboxes;
std::vector<double> deployment_costs, energy_costs, transit_costs, sla_costs,
    total_costs, stretches;
std::vector<double> e_cost_ts;
std::vector<std::vector<int>> ingress_k, egress_k;
std::vector<std::pair<int, int>> num_active_servers;
std::vector<std::vector<double>> sol_closeness;
std::list<int> mbox_count;
std::vector<int> num_service_points;
std::vector<double> net_util;
double per_core_cost, per_bit_transit_cost;
double cost[MAXN][MAXN];
int pre[MAXN][MAXN];
int shortest_path[MAXN][MAXN], sp_pre[MAXN][MAXN];
int shortest_edge_path[MAXN][MAXN];
long bw[MAXN][MAXN];
int max_time;
std::map<std::pair<int, int>, std::unique_ptr<std::vector<int>>> path_cache;
solution_statistics stats;
std::vector<std::unique_ptr<std::vector<int>>> all_results;
std::vector<std::vector<int>> results;
std::vector<std::vector<int>> paths;
middlebox fake_mbox("switch", "0", "0", TOSTRING(INF), "0.0");

int main(int argc, char *argv[]) {
  if (argc < 6) {
    puts(kUsage.c_str());
    return 1;
  }
  auto arg_maps = ParseArgs(argc, argv);
  string algorithm;
  string topology_filename;
  for (auto argument : *arg_maps) {
    if (argument.first == "--per_core_cost") {
      per_core_cost = atof(argument.second.c_str());
    } else if (argument.first == "--per_bit_transit_cost") {
      per_bit_transit_cost = atof(argument.second.c_str());
    } else if (argument.first == "--topology_file") {
      topology_filename = argument.second;
      InitializeTopology(argument.second.c_str());
    } else if (argument.first == "--middlebox_spec_file") {
      InitializeMiddleboxes(argument.second.c_str());
      // PrintMiddleboxes();
    } else if (argument.first == "--traffic_request_file") {
      InitializeTrafficRequests(argument.second.c_str());
    } else if (argument.first == "--algorithm") {
      algorithm = argument.second;
    } else if (argument.first == "--max_time") {
      max_time = atoi(argument.second.c_str());
    }
  }
  if (algorithm == "cplex") {
    std::vector<traffic_request> current_traffic_requests;
    int current_time = traffic_requests[0].arrival_time;
    double opex, running_time;
    int processed_traffic = 0;

    // files to write output
    FILE *cost_log_file = fopen("log.cplex.cost.ts", "w");
    FILE *sequence_log_file = fopen("log.cplex.sequences", "w");
    FILE *path_log_file = fopen("log.cplex.paths", "w");
    FILE *util_log_file = fopen("log.cplex.util.ts", "w");

    // print the node and edge count at the begining of the sequence file
    // fprintf(util_log_file, "%d %d\n", GetNodeCount(graph),
    // GetEdgeCount(graph));

    for (int i = 0; i < traffic_requests.size();) {

      fprintf(cost_log_file, "%d ", current_time);
      fprintf(util_log_file, "%d ", current_time);

      for (; i < traffic_requests.size() &&
                 current_time == traffic_requests[i].arrival_time;
           ++i) {
        current_traffic_requests.push_back(traffic_requests[i]);
        DEBUG("CPLEX arrival time : %d\n", traffic_requests[i].arrival_time);//#
        DEBUG("CPLEX duration time : %d\n", traffic_requests[i].duration);//#
      }
      current_time = traffic_requests[i].arrival_time;

      std::vector<int> sequence[current_traffic_requests.size()];
      std::vector<std::vector<std::pair<int, int>>> edges(
          current_traffic_requests.size());
      std::vector<std::vector<std::pair<int, int>>> all_edges(
          current_traffic_requests.size());
      int delays[current_traffic_requests.size()];
      std::vector<double> opex_breakdown;
      std::vector<int> utilization;

      
      DEBUG("CPLEX input current traffic size : %d\n", current_traffic_requests.size());//#
      run_cplex(current_traffic_requests, opex, opex_breakdown, running_time,
                sequence, edges, all_edges, delays, utilization,
                topology_filename);

      processed_traffic += current_traffic_requests.size();//#
      DEBUG("processed traffic: %d\n", processed_traffic);//#

      // cout << processed_traffic * 100.0 / traffic_requests.size()
      //     << "% Traffic processed." << endl;

      // cost log
      fprintf(cost_log_file, "%lf ", opex);
      for (double cost : opex_breakdown) {
        fprintf(cost_log_file, "%lf ", cost);
      }
      fprintf(cost_log_file, "\n");

      // sequence & path log
      for (int ii = 0; ii < current_traffic_requests.size(); ++ii) {
        // sequence
        std::vector<int> seq = sequence[ii];
        /*
        for (int s : seq) {
          cout << s << " ";
        }
        cout << endl;
        */
        for (int j = 0; j < seq.size(); ++j) {
          fprintf(sequence_log_file, "%d", seq[j]);
          if (j < seq.size() - 1) {
            fprintf(sequence_log_file, ",");
          }
        }
        fprintf(sequence_log_file, "\n");

        // path
        std::vector<std::pair<int, int>> edge_list = edges[ii];
        std::vector<std::pair<int, int>> all_edge_list = all_edges[ii];
        /*
        for (std::pair<int, int> edge: edge_list) {
          cout << "(" << edge.first << ", " << edge.second << ") ";
        }
        cout << endl;
        for (std::pair<int, int> edge: all_edge_list) {
          cout << "(" << edge.first << ", " << edge.second << ") ";
        }
        cout << endl;
        */
        DEBUG("Computing path for traffic %d\n", ii);
        for (auto &edge : edges[ii]) {
          DEBUG("(%d, %d)\n", edge.first, edge.second);
        }
        DEBUG("input sent\n");
        std::vector<int> path = CplexComputePath(edge_list, seq);
        for (int j = 0; j < path.size(); ++j) {
          fprintf(path_log_file, "%d", path[j]);
          if (j < path.size() - 1) {
            fprintf(path_log_file, ",");
          }
        }
        fprintf(path_log_file, "\n");
      }

      /*
      //path log
      for (int t = 0, current, remove_index; t <
      current_traffic_requests.size(); ++t) {
        cout << "processing traffic " << t << endl;
        traffic_request tr = current_traffic_requests[t];
        current = tr.source;
        fprintf(path_log_file, "%d", current);

        std::vector<std::pair <int, int> > pairs = path[t];

        for (std::pair<int, int> p : pairs) {
          cout << "(" << p.first << ", " << p.second << ") ";
        }
        cout << endl;
        cout << "current " << current << endl;

        //while (!pairs.empty()) {
        int loop = 0;
        while (current != tr.destination) {
          loop++;
          remove_index = -1;
          for (int j = 0; j < pairs.size(); ++j) {
            if (current == pairs[j].first) {
              current = pairs[j].second;
              cout << "current " << current << endl;
              fprintf(path_log_file, ",%d", current);
              remove_index = j;
              break;
            }
          }
          //pairs.erase(pairs.begin() + remove_index);
          if (loop == 9)
            break;
        }
        fprintf(path_log_file, "\n");
      }
      */

      // utilization log
      for (int cores : utilization) {
        fprintf(util_log_file, "%d ", cores);
      }
      fprintf(util_log_file, "\n");

      fflush(cost_log_file);
      fflush(sequence_log_file);
      fflush(path_log_file);
      fflush(util_log_file);

      current_traffic_requests.clear();
      // exit(0);
    }

    // close all the output files
    fclose(cost_log_file);
    fclose(sequence_log_file);
    fclose(path_log_file);
    fclose(util_log_file);

  } else if (algorithm == "viterbi") {
    int current_time = traffic_requests[0].arrival_time;
    unsigned long long elapsed_time = 0;
    unsigned long long current_solution_time = 0;
    stats.num_accepted = stats.num_rejected = 0;
    const int kNumTrafficRequests = static_cast<int>(traffic_requests.size());
    for (int i = 0; i < kNumTrafficRequests; ++i) {
      // traffic_requests[i].duration = 6300; // 300;
      if (current_time != traffic_requests[i].arrival_time) {
        // RefreshServerStats(current_time);
        printf("Current time = %d, Solution time = %llu.%llu\n", current_time,
               current_solution_time / ONE_GIG,
               current_solution_time % ONE_GIG);
        current_time = traffic_requests[i].arrival_time;
        current_solution_time = 0;
        ReleaseAllResources();
      }

      // Get solution for one traffic.
      auto solution_start_time = std::chrono::high_resolution_clock::now();
      std::unique_ptr<std::vector<int>> result =
          ViterbiCompute(traffic_requests[i]);
      auto solution_end_time = std::chrono::high_resolution_clock::now();
      unsigned long long solution_time =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              solution_end_time - solution_start_time).count();
      current_solution_time += solution_time;
      elapsed_time += solution_time;
      UpdateResources(result.get(), traffic_requests[i]);
      RefreshServerStats(current_time);
      printf("i = %d, %s\n", i,
      traffic_requests[i].GetDebugString().c_str());
      // Progress bar
      if (i % 500 == 0) {
        double percentage_completed = 100.0 * static_cast<double>(i) /
                                      static_cast<double>(kNumTrafficRequests);
        // printf("%.2lf%% traffics completed\n", percentage_completed);
      }
      all_results.push_back(std::move(result));
    }

    printf("Current time = %d, Solution time = %llu.%llu\n", current_time,
           current_solution_time / ONE_GIG, current_solution_time % ONE_GIG);
    // Print the solution time.
    printf("Solution time: %llu.%llus\n", elapsed_time / ONE_GIG,
           elapsed_time % ONE_GIG);
    printf("Acceptance Ratio: %.8lf%%\n",
           100.0 * static_cast<double>(stats.num_accepted) /
               static_cast<double>(stats.num_accepted + stats.num_rejected));

    // DEBUG: Write all the computed sequences in a file.
    FILE *all_results_file = fopen("log.sequences", "w");
    int row_index = 0;
    for (auto &row : all_results) {
      for (int i = 0; i < row->size(); ++i) {
        if (i != 0) fprintf(all_results_file, ",");
        fprintf(all_results_file, "%d", row->at(i));
      }
      fprintf(all_results_file, "\n");
      ++row_index;
    }
    fclose(all_results_file);
  }
  return 0;
}
