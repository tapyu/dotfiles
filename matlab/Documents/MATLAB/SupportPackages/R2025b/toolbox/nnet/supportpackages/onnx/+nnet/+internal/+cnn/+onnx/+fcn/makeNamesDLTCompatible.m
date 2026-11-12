function graphProto = makeNamesDLTCompatible(graphProto, maxNameLength)
% Rename all nodes, inputs, outputs and initializers to make them
% MATLAB-compatible. Leave empty names empty (e.g., node names).
% maxNameLength defaults to 'namelengthmax'.

%   Copyright 2020-2024 The MathWorks, Inc.    

% Algorithm: 
% 1. Collect all unique names from all subgrphs in the graph tree. 
% 2. Make the names MATLAB-legal using matlab.lang.makeValidName. Note
% that this may have introduced duplicates. 
% 3. Replace names that match dlarray methods. 
% 4. Make the names unique and at most maxNameLength chars using
% matlab.lang.makeUniqueStrings(names, {}, maxNameLength).
% 5. Pair up the original names with the final names using a
% containers.Map.
% 6. Recursively rename all the original names in the graph tree using the
% map.
if nargin < 2
    maxNameLength = namelengthmax;
end
UniqueONNXNames     = uniqueNamesInGraphTree(graphProto);
MATLABNames         = matlab.lang.makeValidName(UniqueONNXNames);
MATLABNames         = replaceDlarrayMethodNames(MATLABNames);
UniqueMATLABNames   = matlab.lang.makeUniqueStrings(MATLABNames, {}, maxNameLength);
NameMap             = containers.Map(UniqueONNXNames, UniqueMATLABNames);
NameMap('')         = '';
graphProto          = replaceNamesInGraphTree(graphProto, NameMap);
end

function Names = uniqueNamesInGraphTree(graphProto)
% Find all the unique names in the graph tree.
% First, collect all names from the current flat graph:
if ~isempty(graphProto.node)
    Names = [{graphProto.name} {graphProto.node.name} [graphProto.node.input] [graphProto.node.output] ...
                graphInputNames(graphProto) graphOutputNames(graphProto)];
else
    Names = [{graphProto.name}, ...
                graphInputNames(graphProto) graphOutputNames(graphProto)];
end
    
if ~isempty(graphProto.initializer)
    Names = [Names {graphProto.initializer.name}];
end
if ~isempty(graphProto.sparse_initializer)
    values = graphProto.sparse_initializer.values;
    Names = [Names {values.name}];
end
% Recursively add names from subgraphs:
nodes = graphProto.node;
for i=1:numel(nodes)
    newNames = namesFromNode(nodes(i));
    if ~isempty(newNames)   % This check is here for performance.
        Names = [Names newNames];
    end
end
% Remove empty names
isEmptyName = cellfun(@isempty, Names);
Names(isEmptyName) = [];
% Get unique names
Names = unique(Names);
end

function Names = replaceDlarrayMethodNames(Names)
ReservedWords = nnet.internal.cnn.onnx.fcn.reservedWords;
for i=1:numel(Names)
    if ismember(Names{i}, ReservedWords)
        Names{i} = nnet.internal.cnn.onnx.fcn.uniqueName(Names{i});
    end
end
end

function names = namesFromNode(nodeProto)
% Recursively find all the names in subgraph trees under this node
names = {};
for i = 1:numel(nodeProto.attribute)
    subgraphs = [nodeProto.attribute(i).g, nodeProto.attribute(i).graphs];
    for g = 1:numel(subgraphs)
        names = [names, uniqueNamesInGraphTree(subgraphs(g))];
    end
end
end

function C = graphInputNames(graphProto)
C = arrayfun(@(valInfoProto)valInfoProto.name, graphProto.input, 'UniformOutput', false);
end

function C = graphOutputNames(graphProto)
C = arrayfun(@(valInfoProto)valInfoProto.name, graphProto.output, 'UniformOutput', false);
end

function graphProto = replaceNamesInGraphTree(graphProto, NameMap)
% Rename names in this graph (non-recursive)
graphProto = renameFlatGraph(graphProto, NameMap);
% Recursively rename names in subgraphs:
for nodeNum = 1:numel(graphProto.node)
    graphProto = renameInsideNode(graphProto, nodeNum, NameMap);
end
end

function graphProto = renameInsideNode(graphProto, NodeNum, NameMap)
% Recursively rename everything in the subgraph tree under node number NodeNum
nodeProto = graphProto.node(NodeNum);
for i = 1:numel(nodeProto.attribute)
    % Find all graphs in this node attibute, rename them, and put them back
    % in the node.
    if ~isempty(nodeProto.attribute(i).g)
        renamedGraph = replaceNamesInGraphTree(nodeProto.attribute(i).g, NameMap);
        nodeProto.attribute(i).g = renamedGraph;
    end
    if ~isempty(nodeProto.attribute(i).graphs)
        renamedGraphs = arrayfun(@(g)replaceNamesInGraphTree(g, NameMap), nodeProto.attribute(i).graphs);
        nodeProto.attribute(i).graphs = renamedGraphs;
    end
    % Put the node back in the graph
    graphProto.node(NodeNum) = nodeProto;
end
end

function graphProto = renameFlatGraph(graphProto, NameMap)
nodes = graphProto.node;
for i=1:numel(nodes)
    % (1) Rename nodes.
    if ~isempty(nodes(i).name)
        nodes(i).name = NameMap(nodes(i).name);
    end
    % (2) Rename node inputs, outputs
    nodes(i).input = arrayfun(@(cellstr)NameMap(cellstr{:}), nodes(i).input, 'UniformOutput', false);
    nodes(i).output = cellfun(@(str)NameMap(str), nodes(i).output, 'UniformOutput', false);
end
graphProto.node = nodes;
% (3) Rename graph, its inputs, outputs and initializers
graphProto.name = NameMap(graphProto.name);
for i = 1:numel(graphProto.input)
    graphProto.input(i).name = NameMap(graphProto.input(i).name);
end
for i = 1:numel(graphProto.output)
    graphProto.output(i).name = NameMap(graphProto.output(i).name);
end
for i = 1:numel(graphProto.initializer)
    graphProto.initializer(i).name = NameMap(graphProto.initializer(i).name);
end
for i = 1:numel(graphProto.sparse_initializer)
    graphProto.sparse_initializer(i).values.name = NameMap(graphProto.sparse_initializer(i).values.name);
end
end
