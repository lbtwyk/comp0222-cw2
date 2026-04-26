function result = optimize_pose_graph_from_mat(sequenceOutputDir, maximumNumberOfOptimizationSteps)
%OPTIMIZE_POSE_GRAPH_FROM_MAT Optimize a prepared Q3d pose graph from MAT input.

if nargin < 2
    maximumNumberOfOptimizationSteps = 50;
end

graphInputPath = fullfile(sequenceOutputDir, 'graph_input.mat');
assert(exist(graphInputPath, 'file') == 2, ...
    'q3d:optimize_pose_graph_from_mat:missinginput', ...
    'Missing graph input file: %s', graphInputPath);

graphInput = load(graphInputPath);

graph = g2o.core.SparseOptimizer();
graph.setAlgorithm(g2o.core.LevenbergMarquardtOptimizationAlgorithm());

numVertices = size(graphInput.keyframe_poses_xytheta, 1);
vertices = cell(numVertices, 1);

for v = 1 : numVertices
    vertex = cw1.drivebot.graph.PlatformStateVertex(graphInput.keyframe_timestamps(v));
    vertex.setEstimate(graphInput.keyframe_poses_xytheta(v, :)');
    if v == graphInput.fixed_keyframe_row
        vertex.setFixed(true);
    end
    graph.addVertex(vertex);
    vertices{v} = vertex;
end

for e = 1 : size(graphInput.odom_edge_vertex_rows, 1)
    edge = q3d.RelativePoseSE2Edge();
    edge.setVertex(1, vertices{graphInput.odom_edge_vertex_rows(e, 1)});
    edge.setVertex(2, vertices{graphInput.odom_edge_vertex_rows(e, 2)});
    edge.setMeasurement(graphInput.odom_edge_measurements(e, :)');
    edge.setInformation(graphInput.odom_edge_information_matrices(:, :, e));
    graph.addEdge(edge);
end

for e = 1 : size(graphInput.loop_edge_vertex_rows, 1)
    edge = q3d.RelativePoseSE2Edge();
    edge.setVertex(1, vertices{graphInput.loop_edge_vertex_rows(e, 1)});
    edge.setVertex(2, vertices{graphInput.loop_edge_vertex_rows(e, 2)});
    edge.setMeasurement(graphInput.loop_edge_measurements(e, :)');
    edge.setInformation(graphInput.loop_edge_information_matrices(:, :, e));
    graph.addEdge(edge);
end

graph.initializeOptimization(false);
chi2Initial = graph.chi2();
iterations = graph.optimize(maximumNumberOfOptimizationSteps);
chi2Final = graph.chi2();

optimizedKeyframePoses = zeros(numVertices, 3);
for v = 1 : numVertices
    optimizedKeyframePoses(v, :) = vertices{v}.estimate()';
end

graphOptimizedPath = fullfile(sequenceOutputDir, 'graph_optimized.mat');
sequence_name = graphInput.sequence_name;
sequence_label = graphInput.sequence_label;
optimized_keyframe_poses_xytheta = optimizedKeyframePoses;
chi2_initial = chi2Initial;
chi2_final = chi2Final;
save(graphOptimizedPath, ...
    'sequence_name', ...
    'sequence_label', ...
    'optimized_keyframe_poses_xytheta', ...
    'chi2_initial', ...
    'chi2_final', ...
    'iterations');

result = struct();
result.sequenceName = sequence_name;
result.sequenceLabel = sequence_label;
result.graphOptimizedPath = graphOptimizedPath;
result.iterations = iterations;
result.chi2Initial = chi2_initial;
result.chi2Final = chi2_final;
end
