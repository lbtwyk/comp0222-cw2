function results = run_all(outputRoot, maximumNumberOfOptimizationSteps)
%RUN_ALL Optimize every prepared Q3d sequence under the output root.

if nargin < 1
    error('q3d:run_all:missingoutputroot', ...
        'Provide the absolute Q3d output root, for example q3d.run_all(''/abs/path/output/lidar_q3d'').');
end

if nargin < 2
    maximumNumberOfOptimizationSteps = 50;
end

graphInputs = dir(fullfile(outputRoot, '**', 'graph_input.mat'));
results = {};
resultIndex = 0;

for idx = 1 : numel(graphInputs)
    graphInput = graphInputs(idx);
    graphInputPath = fullfile(graphInput.folder, graphInput.name);
    if contains(graphInputPath, [filesep 'variants' filesep]) == false
        continue;
    end

    sequenceOutputDir = graphInput.folder;
    fprintf('Optimizing %s\n', sequenceOutputDir);
    resultIndex = resultIndex + 1;
    results{resultIndex, 1} = q3d.optimize_pose_graph_from_mat( ...
        sequenceOutputDir, maximumNumberOfOptimizationSteps);
end
end
