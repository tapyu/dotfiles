classdef GraphTranslation < handle      % A handle so we can put a pointer to it inside NodeTranslation

    %   Copyright 2020-2026 The MathWorks, Inc.    

    properties
        GraphName
        
        % The GraphProto being translated
        GraphProto
        
        % We define "External" input and output tensors to be those that
        % serve as the "I/O" of the graph as a whole. This differs from
        % what ONNX calls the graph "inputs", which also includes purely
        % graph-internal tensors that appear as inputs to operators, for
        % example, weight tensors. Operationally, we define external input
        % tensors to be ONNX inputs that do not have corresponding
        % initializers.
        ExternalInputNames
        ExternalInputShapes
        ExternalInputTypes
        ExternalInputRanks
        
        % ONNX graph "outputs" are already external.
        ExternalOutputNames
        ExternalOutputShapes
        ExternalOutputTypes
        
        % An array of translations of nodes in this graph
        NodeTranslations = [];
        
        % A struct of initial values for graph inputs, including internal
        % inputs, to be used when an input is not passed.
        AllInputDefaults = struct;
        
        % A struct of new state properties with their initial values that
        % are introduced in this graph and its subgraphs. (Currently, this
        % is only batchnorm running means and variances).
        InitialState = struct;
        
        % A struct of learnable parameters with their initial values that
        % are introduced in this graph and its subgraphs.
        Learnables = struct;
        
        % A struct of nonlearnable parameters with their values that are
        % introduced in this graph and its subgraphs.
        Nonlearnables = struct;
        
        % Names of onnx<OP> functions required for this graph and
        % subgraphs. A string array.
        IncludedFunctionNames = string([]);
        
        % MATLAB code for functions implementing this graph and all
        % subgraphs. A string array.
        GraphFunctionCode = string([]);
        
        % The opset version of the model that contains this graph
        OpsetVersion = 1;
        
        % The names of all integer tensors. A string array.
        IntegerTensorNames = string([]);
        
        % A list of NodeTranslationIssues in this graph and all subgraphs.
        TranslationIssues = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
        
        % The Namespace name
        PackageName;

        % Intermediate Tensor names
        IntermediateTensorNames = string([]);
        ParentsIntermediateTensorNames = string([]);

        % Handle to top level graph translation
        TopLevelGraphTranslation;

        % Whether to generate C-codegen compatible translations.
        CoderMode
        
        % Index of ONNX node being translated now
        NodeTranslationIdx
    end
    
    methods
        function this = GraphTranslation(graphProto, OpsetVersion, initialIntegerTensorNames, PackageName, TopLevelGraphTranslation, ...
                                           doCoderCompatibleTranslation, optionalGraphName)
            if nargin < 5
                TopLevelGraphTranslation = [];
                doCoderCompatibleTranslation = false;
                optionalGraphName = nnet.internal.cnn.onnx.fcn.uniqueName([graphProto.name 'Graph']);
            elseif nargin < 6
                doCoderCompatibleTranslation = false;
                optionalGraphName = nnet.internal.cnn.onnx.fcn.uniqueName([graphProto.name 'Graph']);
            elseif nargin < 7
                optionalGraphName = nnet.internal.cnn.onnx.fcn.uniqueName([graphProto.name 'Graph']);
            end
            this.CoderMode = doCoderCompatibleTranslation;
            this.GraphProto = graphProto;
            this.OpsetVersion = OpsetVersion;
            this.PackageName = PackageName;
            this.IntegerTensorNames = initialIntegerTensorNames;        % Integer tensor names form an enclosing scope (if exists).
            this.GraphName = optionalGraphName;
            this = findGraphIOTensors(this, graphProto);
            this.IntegerTensorNames	= [this.IntegerTensorNames, integerInitializerNames(this, graphProto)];
            this.TopLevelGraphTranslation = TopLevelGraphTranslation;
            
            % If the graph is a subgraph, inherits all tensor information
            % from its parents
            if ~isempty(TopLevelGraphTranslation)
                this.Nonlearnables = TopLevelGraphTranslation.Nonlearnables;
                this.Learnables = TopLevelGraphTranslation.Learnables;
                this.InitialState = TopLevelGraphTranslation.InitialState;
                this.ParentsIntermediateTensorNames = [TopLevelGraphTranslation.ExternalInputNames TopLevelGraphTranslation.IntermediateTensorNames];
            end

            % Translate the ONNX nodes
            this.NodeTranslations = nnet.internal.cnn.onnx.fcn.NodeTranslation.empty;
            for i = 1:numel(graphProto.node)
                % Store the index in NodeTranslationIdx so that 
                % we can use it in getVarAndNumDimName to access 
                % the node inputs and generate codegenable code that
                % facilitates coder constant folding
                this.NodeTranslationIdx = i;
                this.NodeTranslations(i) = nnet.internal.cnn.onnx.fcn.NodeTranslation.create(graphProto.node(i), this.OpsetVersion, this.IntegerTensorNames, this);
                % Update list of integer tensors after each node is
                % translated, for use by subsequent nodes
                this.IntegerTensorNames = [this.IntegerTensorNames, this.NodeTranslations(i).IntegerOutputTensorNames(:)'];
            end

            % Collect subgraph learnables, nonlearnables and included function names from nodes
            if ~isempty(TopLevelGraphTranslation)
                this.TopLevelGraphTranslation.Nonlearnables = this.Nonlearnables;
                this.TopLevelGraphTranslation.Learnables = this.Learnables;
                this.TopLevelGraphTranslation.InitialState = this.InitialState;
            end

            if ~isempty([this.NodeTranslations.IncludedFunctionNames])
                this.IncludedFunctionNames = unique([this.NodeTranslations.IncludedFunctionNames]);
            end

            % Generate code to implement the graph.
            this.GraphFunctionCode = genGraphDefinition(this);
            % Append any subgraph function code from the nodes.
            SubgraphTranslations = [this.NodeTranslations.SubgraphTranslations];
            if ~isempty(SubgraphTranslations)
                this.GraphFunctionCode = [this.GraphFunctionCode, SubgraphTranslations.GraphFunctionCode];
            end
            % Collect all NodeTranslationIssues
            issues = arrayfun(@(nodeTranslation) nodeTranslation.Issues, this.NodeTranslations, 'UniformOutput', false);
            this.TranslationIssues = [issues{:}];
        end
        
        function tf = isInitializer(this, name)
            tf = ~isempty(this.GraphProto.sparse_initializer) && ismember(name, {this.GraphProto.sparse_initializer.name}) ...
                || ~isempty(this.GraphProto.initializer) && ismember(name, {this.GraphProto.initializer.name}) ...
                || ~isempty(this.TopLevelGraphTranslation) && isInitializer(this.TopLevelGraphTranslation,name);
        end  

        function tf = isInitializerThisGraph(this, name)
            tf = ~isempty(this.GraphProto.sparse_initializer) && ismember(name, {this.GraphProto.sparse_initializer.name}) ...
                || ~isempty(this.GraphProto.initializer) && ismember(name, {this.GraphProto.initializer.name});
        end 
        
        function [data, rankedArray] = getInitializerValue(this, name)
            isSparse = false;
            if isInitializerThisGraph(this, name)
                if ~isempty(this.GraphProto.sparse_initializer)
                    [isSparse,idx] = ismember(name, {this.GraphProto.sparse_initializer.name});
                end
                if isSparse
                    initializer = this.GraphProto.sparse_initializer(idx);
                    data = nnet.internal.cnn.onnx.getDataFromSparseTensorProto(initializer);
                else
                    [~,idx] = ismember(name, {this.GraphProto.initializer.name});
                    initializer = this.GraphProto.initializer(idx);
                    data = nnet.internal.cnn.onnx.getDataFromTensorProto(initializer);
                end
            else
                [data, rankedArray] = getInitializerValue(this.TopLevelGraphTranslation, name);
                return
            end

            % Make numeric data double so that input doubles can remain
            % doubles all the way through
            if isnumeric(data) && ~isa(data, 'double')
                data = double(data);
            end
            % Get dims
            dimVec = flip(initializer.dims);                                    % flip initializer dims to MATLAB dimension ordering.
            if numel(dimVec) > 1
                dimVec = dimVec(:)';                                            % Make sure dimVec is a row vector.
                data = reshape(data, dimVec);                                   % Apply MATLAB shape
            end
            rankedArray = nnet.internal.cnn.onnx.fcn.RankedArray(data, numel(dimVec));  % Add rank info
        end
        
        function [VarName, NumDimName] = getVarAndNumDimName(this, name)
            if ~isempty(this.PackageName)
                if isfield(this.Learnables, name)
                    % If the variable is a learnable
                    VarName = ['this.', name];
                    NumDimName = ['this.NumDims.', name];
                elseif isfield(this.InitialState, name)
                    % If the variable is a state, store it in Var for
                    % code-gen. Store it as state in Vars.
                    if this.CoderMode
                        VarName = ['this.Vars.', name];
                    else
                        VarName = ['this.', name];
                    end
                    NumDimName = ['this.NumDims.', name];
                elseif isfield(this.Nonlearnables, name)
                    % If the variable is a non-learnable
                    VarName = ['this.Vars.', name];
                    NumDimName = ['this.NumDims.', name];
                elseif isInitializer(this, name)
                    % If the variable is in initializer but not set as
                    % non-learnable or learnable, set it as a learnable
                    % as we always assign properties to non-learnables when
                    % needed. 
                    [~, rankedArray] = getInitializerValue(this, name);
                    % store it in Nonlearnables. 
                    this.Nonlearnables.(name) = rankedArray;
                    VarName = ['this.Vars.', name];
                    NumDimName = ['this.NumDims.', name];
                    
                else
                    % If the variable is not a property
                    VarName = name;
                    NumDimName = [name, 'NumDims'];
                    if ~ismember(name, this.ExternalInputNames)
                        this.IntermediateTensorNames  = [this.IntermediateTensorNames name];
                    end
                    if this.CoderMode && ismember(name, this.GraphProto.node(this.NodeTranslationIdx).input)
                        % Add coder.const to NumDimName if it 
                        % corresponds to a node input, to help 
                        % trigger coder constant folding
                        NumDimName = ['coder.const(', NumDimName, ')'];
                    end
                end
            else
                % If called from importONNXFunction, parameter names
                % are always stored in Vars and numdims are always stored
                % in NumDims struct. 
                VarName = ['Vars.', name];
                NumDimName = ['NumDims.', name];
                if isInitializer(this, name) && ...
                    ~isfield(this.Nonlearnables, name) && ...
                    ~isfield(this.InitialState, name) && ...
                    ~isfield(this.Learnables, name)
                    % If the variable is a initializer but not set as
                    % initial state, learnables or nonlearnables
                    [~, rankedArray] = getInitializerValue(this, name);
                    this.Nonlearnables.(name) = rankedArray;
                end
            end
        end
        function tf = setInitializerType(this, type, name)
            % if 'name' is a initializer, set it as type
            tf = false;
            if isInitializer(this, name)
                [~, rankedArray] = getInitializerValue(this, name);
                tf = setParameters(this, type, name, rankedArray);
            end
        end
        function tf = setParameters(this, paramType, paramName, value)
            if ~isfield(this.Nonlearnables, paramName) && ...
                ~isfield(this.InitialState, paramName) && ...
                ~isfield(this.Learnables, paramName)
                tf = true;
                this.(paramType).(paramName) = value;
            else
                tf = false;
            end
        end
        
        function packageName = getPackageName(this)
            if ~isempty(this.PackageName)
                if this.CoderMode
                    packageName = [this.PackageName, '.coder.ops.'];
                else
                    packageName = [this.PackageName, '.ops.'];
                end                
            else
                packageName = '';
            end
        end
    end
    
    methods(Access=protected)
        function code = genGraphDefinition(this)
            if isempty(this.PackageName)
                 % Write the function header line for importONNXFunction
                InputRankStrings = cellfun(@(v)nnet.internal.cnn.onnx.fcn.uniqueName(sprintf('%sNumDims', v)), this.ExternalInputNames, 'UniformOutput', false);
                AllInputStrings = [this.ExternalInputNames, InputRankStrings {'Vars', 'NumDims', 'Training', 'state'}];
                OutputRankStrings = cellfun(@(v)nnet.internal.cnn.onnx.fcn.uniqueName(sprintf('%sNumDims', v)), this.ExternalOutputNames, 'UniformOutput', false);
                AllOutputStrings = [this.ExternalOutputNames, OutputRankStrings, {'state'}];
                code = sprintf('function [%s] = %s(%s)\n', strjoin(AllOutputStrings, ', '), this.GraphName, strjoin(AllInputStrings, ', '));
                code = [code sprintf('%% Function implementing the graph ''%s''\n', this.GraphName)];
                % Set Vars fields from formal input parameters.
                code = [code sprintf('%% Update Vars and NumDims from the graph''s formal input parameters. Note that state variables are already in Vars.\n')];
                for i = 1:numel(this.ExternalInputNames)
                    name = this.ExternalInputNames{i};
                    code = [code sprintf('Vars.%s = %s;\n', name, name)];
                    code = [code sprintf('NumDims.%s = %s;\n', name, InputRankStrings{i})];
                end
                % Insert calls to operators
                code = [code newline sprintf('%% Execute the operators:\n')];
                MCode = [this.NodeTranslations.MCode];
                for i = 1:numel(MCode)
                    code = [code sprintf('%s', MCode{i}) newline];
                end
                % Set output arguments from Vars and NumDims
                code = [code sprintf('%% Set graph output arguments from Vars and NumDims:\n')];
                for i = 1:numel(this.ExternalOutputNames)
                    name = this.ExternalOutputNames{i};
                    code = [code sprintf('%s = Vars.%s;\n', name, name)];
                    code = [code sprintf('%s = NumDims.%s;\n', OutputRankStrings{i}, name)];
                end
                % Set output state from Vars
                code = [code sprintf('%% Set output state from Vars:\n')];
                code = [code sprintf('state = updateStruct(state, Vars);\n')];
                % End function definition
                code = [code 'end' newline];
                code = string(code);
            else
                % Write the function for importONNXLayer,
                % importNetworkFromONNX and importONNXNetwork
                InputArgs = ['this', this.ExternalInputNames, cellfun(@(ch)[ch 'NumDims'], this.ExternalInputNames, 'UniformOutput',false)];
                ParentsTensors = intersect(this.ParentsIntermediateTensorNames, this.IntermediateTensorNames);
                if ~isempty(ParentsTensors)
                    InputArgs = [InputArgs, ParentsTensors, cellfun(@(ch)[ch 'NumDims'], ParentsTensors, 'UniformOutput',false)];
                end
                InputArgs  = [InputArgs  'Training'];
                OutputRankStrings = cellfun(@(v)nnet.internal.cnn.onnx.fcn.uniqueName(sprintf('%sNumDims', v)), this.ExternalOutputNames, 'UniformOutput', false);
                OutputArgs = [this.ExternalOutputNames, OutputRankStrings, fieldnames(this.InitialState)'];

                % Generate function header for this graph
                code = sprintf('function [%s] = %s(%s)\n', strjoin(OutputArgs, ', '), this.GraphName, strjoin(InputArgs, ', '));

                code = [code newline sprintf('%% Execute the operators:\n')];
                MCode = [this.NodeTranslations.MCode];
                for i = 1:numel(MCode)
                    code = [code sprintf('%s', MCode{i}) newline];
                end
                % Set output arguments from Vars and NumDims
                setOutputCode = this.genSetOutputCode(OutputRankStrings);
                if ~isempty(setOutputCode)
                    code = [code sprintf('%% Set graph output arguments \n')];
                    code = [code sprintf('%s', setOutputCode)];
                end                    
                code = [code "end" newline];
                code = string(code);
            end
        end

        function code = genSetOutputCode(this, OutputRankStrings)
            code ='';
            % If the output names are different from the names used inside
            % the graph, generates code to assign output names the values. 
            for i = 1:numel(this.ExternalOutputNames)
                name = this.ExternalOutputNames{i};
                [varName,numDimName] = getVarAndNumDimName(this, name);
                if ~isequal(name,varName)
                    code = [code sprintf('%s = %s;\n', name, varName)];
                end
                if ~isequal(OutputRankStrings{i},numDimName)
                    if this.CoderMode
                        code = [code sprintf('%s = coder.const(%s);\n', OutputRankStrings{i},numDimName)];
                    else
                        code = [code sprintf('%s = %s;\n', OutputRankStrings{i},numDimName)];
                    end
                end
            end
            initialStateName = fieldnames(this.InitialState);
            for i = 1:numel(initialStateName)
                name = initialStateName{i};
                [varName,~] = getVarAndNumDimName(this, name);
                if ~isequal(name,varName)
                    code = [code sprintf('%s = %s;\n', name, varName)];
                end
            end
        end
        
        function nameStrings = integerInitializerNames(~, graphProto)
            if isempty(graphProto.initializer)
                nameStrings = string([]);
            else
                initializerNames = {graphProto.initializer.name};
                initializerTypes = {graphProto.initializer.data_type};
                isInt = cellfun(@nnet.internal.cnn.onnx.isONNXTypeInteger, initializerTypes);
                nameStrings = string(initializerNames(isInt));
            end
        end
                
        function this = findGraphIOTensors(this, graphProto)
            % Find the input and output tensors of the graph as a whole and
            % record their properties
            GraphInputs = allGraphInputNames(graphProto);
            inputsWithoutInitializers = GraphInputs;
            if ~isempty(graphProto.initializer)
                inputsWithoutInitializers = setdiff(inputsWithoutInitializers, {graphProto.initializer.name}, 'stable');     % Stable keeps the inputs in the correct order.
            end
            this.ExternalInputNames   = inputsWithoutInitializers;
            this.ExternalOutputNames  = graphOutputNames(graphProto);
            [this.ExternalInputShapes, this.ExternalInputTypes]   = findGraphInputShapesAndTypes(graphProto, this.ExternalInputNames);
            [this.ExternalOutputShapes, this.ExternalOutputTypes]	= findGraphOutputShapesAndTypes(graphProto, this.ExternalOutputNames);
            % Create ExternalInputRanks struct. Add input tensors.
            this.ExternalInputRanks = struct;
            for i = 1:numel(this.ExternalInputNames)
                if isempty(this.ExternalInputShapes{i})
                    this.ExternalInputRanks.(this.ExternalInputNames{i}) = [];                                  % Empty shapes get empty rank
                else
                    this.ExternalInputRanks.(this.ExternalInputNames{i}) = numel(this.ExternalInputShapes{i});
                end
            end
            % Add integer input tensors
            for i = 1:numel(this.ExternalInputNames)
                if nnet.internal.cnn.onnx.isONNXTypeInteger(this.ExternalInputTypes{i})
                    this.IntegerTensorNames(end+1) = this.ExternalInputNames{i};
                end
            end
        end        
    end
end

function [graphInputShapes, graphInputTypes] = findGraphInputShapesAndTypes(graphProto, inputTensorNames)
graphInputShapes = {};
graphInputTypes = string;
% Find the input tensors
[~,positions] = ismember(inputTensorNames, allGraphInputNames(graphProto));
assert(all(positions>0));   % Meaning the input tensors were found
for i = 1:numel(positions)
    pos = positions(i);
    % Get the shape cell array for this input. Fill in empty cells with the sizes specified.
    shape = graphProto.input(pos).type.tensor_type.shape;
    if isempty(shape) || isempty(shape.dim)
        ONNXInputShape      = {};
        ONNXInputSymbols    = {};
    else
        ONNXInputShape      = cellfun(@double, {graphProto.input(pos).type.tensor_type.shape.dim.dim_value}, 'UniformOutput', false);
        ONNXInputSymbols	= {graphProto.input(pos).type.tensor_type.shape.dim.dim_param};          % Non-empty elements indicate undetermined sizes.
        [ONNXInputShape, ONNXInputSymbols] = addMissingShapeSymbols(ONNXInputShape, ONNXInputSymbols);
    end
    ONNXShapeCell       = ONNXInputSymbols;
    isEmptyShape        = cellfun(@isempty, ONNXShapeCell);
    ONNXShapeCell(isEmptyShape) = ONNXInputShape(isEmptyShape);
    graphInputShapes{i} = ONNXShapeCell;
    % Get the type for this input.
    graphInputTypes(i) = graphProto.input(pos).type.tensor_type.elem_type;
end
end

function C = allGraphInputNames(graphProto)
% All "input" names to the graph, including both "true" external inputs as
% well as internal tensors that have initializers.
C = arrayfun(@(valInfoProto)valInfoProto.name, graphProto.input, 'UniformOutput', false);
end

function C = graphOutputNames(graphProto)
C = arrayfun(@(valInfoProto)valInfoProto.name, graphProto.output, 'UniformOutput', false);
end

function [ONNXShape, ONNXSymbols] = addMissingShapeSymbols(ONNXShape, ONNXSymbols)
% Add symbols for entries that have neither a shape nor a symbol
persistent newSymbolNum
for j = 1:numel(ONNXSymbols)
    if isempty(ONNXSymbols{j}) && isempty(ONNXShape{j})
        if isempty(newSymbolNum)
            newSymbolNum = 1;
        else
            newSymbolNum = newSymbolNum + 1;
        end
        ONNXSymbols{j} = sprintf('Unknown%d', newSymbolNum);
    end
end
end

function [graphOutputShapes, graphOutputTypes] = findGraphOutputShapesAndTypes(graphProto, outputTensorNames)
graphOutputShapes = {};
graphOutputTypes = string;
% Find the output tensors
[~,positions] = ismember(outputTensorNames, graphOutputNames(graphProto));
assert(all(positions>0));   % Meaning the output tensors were found
for i = 1:numel(positions)
    pos = positions(i);
    % Get its shape cell array. Fill in empty cells with the sizes
    % specified. From the ONNX IR Doc: "The empty list of dimension sizes,
    % [], is a valid tensor shape, denoting a zero-dimension (scalar)
    % value."
    if isempty(graphProto.output(pos).type)
        % This condition occurs in ONNX networks we've created when we add
        % all intermediate tensors to the output for debugging purposes.
        graphOutputShapes{i} = {};
        graphOutputTypes(i) = nnet.internal.cnn.onnx.TensorProto_DataType.FLOAT;
    else
        if isempty(graphProto.output(pos).type.tensor_type)
        % Only tensor outputs can be represented as a dlnetwork
        % layer output. Sequence/map/optional/etc. output types (or
        % a missing type altogether) are unsupported. Error clearly,
        % naming the output and its type, instead of crashing on the
        % missing tensor_type.
        error(message('nnet_cnn_onnx:onnx:UnsupportedModelOutputDataType', ...
            graphProto.output(pos).name, iOutputTypeName(graphProto.output(pos).type)));
        end
        shape = graphProto.output(pos).type.tensor_type.shape;
        if isempty(shape) || isempty(shape.dim)
            % This condition occurs in ONNX models in the wild
            graphOutputShapes{i} = {};
        else
            dim = shape.dim;
            ONNXOutputShape  	= cellfun(@double, {dim.dim_value}, 'UniformOutput', false);
            ONNXOutputSymbols	= {dim.dim_param};                                  % Non-empty elements indicate undetermined sizes.
            [ONNXOutputShape, ONNXOutputSymbols] = addMissingShapeSymbols(ONNXOutputShape, ONNXOutputSymbols);
            ONNXShapeCell       = ONNXOutputSymbols;
            isEmptyShape        = cellfun(@isempty, ONNXShapeCell);
            ONNXShapeCell(isEmptyShape) = ONNXOutputShape(isEmptyShape);
            graphOutputShapes{i} = ONNXShapeCell;
        end
        % Get the type for this output.
        graphOutputTypes(i) = graphProto.output(pos).type.tensor_type.elem_type;
    end
end

end
function name = iOutputTypeName(type)
% Return the name of the ONNX output TypeProto's set type, i.e. the
% populated field of the "value" oneof (tensor_type, sequence_type, ...).
% Used to report unsupported non-tensor output types. Any new oneof field
% added to TypeProto is picked up automatically; only 'denotation' is
% skipped since it is a plain string sibling, not part of the oneof.
name = "unknown";
if isempty(type)
    return
end
typeFields = string(fieldnames(type));
typeFields(typeFields == "denotation") = [];
for i = 1:numel(typeFields)
    fName = typeFields(i);
    if ~isempty(type.(fName))
        % Turn e.g. "sequence_type" into "sequence" for the message.
        name = erase(fName, "_type");
    end
end
end