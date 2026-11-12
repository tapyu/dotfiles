classdef ModelTranslation

    %   Copyright 2020-2025 The MathWorks, Inc.    

    properties
        % Output function path
        OutputFcnPath
        
        % Output function name
        OutputFcnName
        
        % The translation of this model's top-level graph
        ToplevelGraphTranslation

        % The C-codegen compatible translation of this model's top-level graph
        ToplevelGraphCoderTranslation
        
        % A ONNXParameters object
        Params
        
        % Complete MATLAB code implementing this model. A string array.
        ModelFunctionCode = [];

        % Complete C-codegen compatible MATLAB code implementing this model. A string array.
        ModelFunctionCoderCode = [];
        
        % The opset version of the model
        OpsetVersion = 1

        % The Namespace of the model
        PackageName
    end
    
    methods
        function this = ModelTranslation(modelProto, outputFcnPath, outputFcnName, maxNameLength, packageName, isGeneratingCustomLayerOPTIONAL, generateCoderCode)
            if nargin < 6
                generateCoderCode = false;
                isGeneratingCustomLayerOPTIONAL = false;
            elseif nargin < 7
                generateCoderCode = false;
            end            
            this.OutputFcnPath = outputFcnPath;
            this.OutputFcnName = outputFcnName;
            this.PackageName = packageName;
            % Rename all the graph variables
            modelProto.graph = nnet.internal.cnn.onnx.fcn.makeNamesDLTCompatible(modelProto.graph, maxNameLength);
            % Find opset version
            if ~isempty(modelProto.opset_import)
                this.OpsetVersion = max(double([modelProto.opset_import.version]));
            else
                this.OpsetVersion = 1;
            end
            this.ToplevelGraphTranslation = nnet.internal.cnn.onnx.fcn.GraphTranslation(modelProto.graph, this.OpsetVersion, "", this.PackageName, [], generateCoderCode);
            this.Params                   = fillONNXParameters(this);
            this.ModelFunctionCode        = genOutputFcn(this, outputFcnName, this.ToplevelGraphTranslation, isGeneratingCustomLayerOPTIONAL);           
        end
        
        function params = fillONNXParameters(this)
            % Create ONNXParameters object and add all parameters
            params   = ONNXParameters.createONNXParameters(this.OutputFcnName);
            State    	= this.ToplevelGraphTranslation.InitialState;
            Learnables	= this.ToplevelGraphTranslation.Learnables;
            Nonlearnables	= this.ToplevelGraphTranslation.Nonlearnables;
            names       = fieldnames(State);
            for i=1:numel(names)
                val = State.(names{i});
                params = params.addParameter(names{i}, val.Array, 'state', val.Rank);
            end
            names = fieldnames(Nonlearnables);
            for i=1:numel(names)
                val = Nonlearnables.(names{i});
                params = params.addParameter(names{i}, val.Array, 'nonlearnable', val.Rank);
            end
            names = fieldnames(Learnables);
            for i=1:numel(names)
                val = Learnables.(names{i});
                params = params.addParameter(names{i}, val.Array, 'learnable', val.Rank);
            end
        end
        
        function writeOutputFile(this)
            outputFilename = fullfile(this.OutputFcnPath, [this.OutputFcnName '.m']);
            % Save code to file
            if isfile(outputFilename)
                try
                    delete(outputFilename);
                catch me
                    error(message('nnet_cnn_onnx:onnx:FileOverwrite', outputFilename, me.message));
                end
            end
            [f, errmsg] = fopen(outputFilename, 'w');
            if f < 0
                error(message('nnet_cnn_onnx:onnx:FileCreation', outputFilename, errmsg));
            end
            fprintf(f, '%s', this.ModelFunctionCode);
            fclose(f);
            rehash;
        end
        
        function code = genOutputFcn(this, outputFcnName, graphTranslation, isGeneratingCustomLayer)
            if ~isGeneratingCustomLayer
                % Write the code for importONNXFunction
                code = '';
                code = [code genFunctionHeaderLine(graphTranslation, outputFcnName)];       % The function header line
                code = [code genHelpText(this, graphTranslation, outputFcnName)];                 % Help text
                code = [code genCallPreprocessInput(graphTranslation)];
                code = [code genPackageVariables(graphTranslation, isGeneratingCustomLayer)];                      	% Fill Vars and NumDims
                code = [code genCallTopLevelGraph(graphTranslation)];                       % Call the top-level graph function
                code = [code genCallPostprocessOutput(graphTranslation)];
                code = [code sprintf('end\n\n')];                                          	% End the main function definition
                code = [code char(join(graphTranslation.GraphFunctionCode,newline))]; % Append the top-level graph function definition
                code = [code newline];
                code = [code genInputValidationFunction(graphTranslation, outputFcnName)];  % Append helper function to validate inputs
                code = [code genPreprocessInputFunction(graphTranslation, isGeneratingCustomLayer)];
                code = [code genPostprocessOutputFunction(graphTranslation)];
                code = [code genIncludedFunctionDefinitions(graphTranslation.IncludedFunctionNames)];
                code = [code genSpkgUtilityFunctions()];
                code = nnet.internal.cnn.onnx.util.indentcode(code);                                                    % Indent the code                                                % Indent the code
            else
                % Write the code for importONNXNetwork,
                % importNetworkFromONNX and importONNXLayer workflow
                code = '';
                code = [code char(join(graphTranslation.GraphFunctionCode,newline))];
                code = [code sprintf('end\n')];
                code = nnet.internal.cnn.onnx.util.indentcode(code);
            end
        end

        function code = genHelpText(this, graphTranslation, outputFcnName)
            code = sprintf('%%%s Function implementing an imported ONNX network. \n', upper(outputFcnName));
            code = [code genCommentedNewline];
            code = [code sprintf('%% THIS FILE WAS AUTO-GENERATED BY importONNXFunction.\n')];
            code = [code sprintf('%% ONNX Operator Set Version: %d\n', this.OpsetVersion)];
            code = [code genCommentedNewline];
            code = [code sprintf('%% Variable names in this function are taken from the original ONNX file.\n')];
            code = [code genCommentedNewline];
            OutputArgs = strjoin(graphTranslation.ExternalOutputNames, ', ');
            OutputArgsWithState = strjoin([OutputArgs, {'state'}], ', ');
            InputArgs = strjoin(graphTranslation.ExternalInputNames, ', ');
            
            % Function header and description
            code = [code sprintf('%% [%s] = %s(%s, PARAMS)\n', upper(OutputArgs), outputFcnName, upper(InputArgs))];
            code = [code sprintf('%%\t\t\t- Evaluates the imported ONNX network %s with input(s)\n', upper(outputFcnName))];
            code = [code sprintf('%%\t\t\t%s and the imported network parameters in PARAMS. Returns\n',  upper(InputArgs))];
            code = [code sprintf('%%\t\t\tnetwork output(s) in %s.\n', upper(OutputArgs))];
            code = [code genCommentedNewline];
            code = [code sprintf('%% [%s] = %s(%s, PARAMS)\n', upper(OutputArgsWithState), outputFcnName, upper(InputArgs))];
            code = [code sprintf('%%\t\t\t- Additionally returns state variables in STATE. When training,\n')];
            code = [code sprintf('%%\t\t\tuse this form and set TRAINING to true.\n')];
            code = [code genCommentedNewline];
            code = [code sprintf('%% [__] = %s(%s, PARAMS, ''NAME1'', VAL1, ''NAME2'', VAL2, ...)\n', outputFcnName, upper(InputArgs))];
            code = [code sprintf('%%\t\t\t- Specifies additional name-value pairs described below:\n')];
            code = [code genCommentedNewline];
            code = [code sprintf('%% ''Training''\n')];
            code = [code sprintf('%% \t\t\tBoolean indicating whether the network is being evaluated for \n')];
            code = [code sprintf('%%\t\t\tprediction or training. If TRAINING is true, state variables\n')];
            code = [code sprintf('%%\t\t\twill be updated.\n')];
            code = [code genCommentedNewline];
            code = [code sprintf('%% ''InputDataPermutation''\n')];
            code = [code sprintf(['%%\t\t\t''auto'' - Automatically attempt to determine the permutation\n'...
                '%%\t\t\t between the dimensions of the input data and the dimensions of \n'...
                '%%\t\t\tthe ONNX model input. For example, the permutation from HWCN \n'...
                '%%\t\t\t(MATLAB standard) to NCHW (ONNX standard) uses the vector \n'...
                '%%\t\t\t[4 3 1 2]. See the documentation for IMPORTONNXFUNCTION for \n' ...
                '%%\t\t\tmore information about automatic permutation.\n'])];
            code = [code genCommentedNewline];
            code = [code sprintf('%%\t\t\t''none'' - Input(s) are passed in the ONNX model format. See ''Inputs''.\n')];
            code = [code genCommentedNewline];
            code = [code sprintf(['%%\t\t\tnumeric vector - The permutation vector describing the \n'...
                '%%\t\t\ttransformation between input data dimensions and the expected \n'...
                '%%\t\t\tONNX input dimensions.'])];
            code = [code genCommentedNewline];
            code = [code sprintf(['%%\t\t\tcell array - If the network has multiple inputs, each cell \n'...
                '%%\t\t\tcontains ''auto'', ''none'', or a numeric vector.\n'])];
            code = [code genCommentedNewline];
            code = [code sprintf('%% ''OutputDataPermutation''\n')];
            code = [code sprintf(['%%\t\t\t''auto'' - Automatically attempt to determine the permutation \n'...
                '%%\t\t\tbetween the dimensions of the output and a conventional MATLAB \n'...
                '%%\t\t\tdimension ordering. For example, the permutation from NC (ONNX \n'...
                '%%\t\t\tstandard) to CN (MATLAB standard) uses the vector [2 1]. See \n'...
                '%%\t\t\tthe documentation for IMPORTONNXFUNCTION for more information \n'...
                '%%\t\t\tabout automatic permutation.\n'])];
            code = [code genCommentedNewline];
            code = [code sprintf('%%\t\t\t''none'' - Return output(s) as given by the ONNX model. See ''Outputs''.\n')];
            code = [code genCommentedNewline];
            code = [code sprintf(['%%\t\t\tnumeric vector - The permutation vector describing the\n'...
                '%%\t\t\ttransformation between the ONNX output dimensions and the  \n'...
                '%%\t\t\tdesired output dimensions.'])];
            code = [code genCommentedNewline];
            code = [code sprintf(['%%\t\t\tcell array - If the network has multiple outputs, each cell \n'...
                '%%\t\t\tcontains ''auto'', ''none'' or a numeric vector.\n'])];
            code = [code genCommentedNewline];
            
            % Required Inputs
            code = [code sprintf('%% Inputs:\n%% -------\n')];
            code = [code sprintf('%% %s\n', upper(InputArgs))];
            code = [code sprintf('%%\t\t\t- Input(s) to the ONNX network.\n')];
            code = [code sprintf('%%\t\t\t  The input size(s) expected by the ONNX file are:\n')];
            for i=1:numel(graphTranslation.ExternalInputNames)
                shape = graphTranslation.ExternalInputShapes{i};
                inputType = graphTranslation.ExternalInputTypes{i};
                if isempty(shape)
                    shape = {1,1};
                elseif numel(shape)==1
                    shape{2} = 1;
                end
                inputSizes = "[" + strjoin(cellfun(@string, shape), ", ") + "]";
                code = [code sprintf('%%\t\t\t\t  %s:\t\t%s\t\t\t\tType: %s\n', upper(graphTranslation.ExternalInputNames{i}), inputSizes, inputType)];
            end
            code = [code sprintf('%%\t\t\t  By default, the function will try to permute the input(s) \n')];
            code = [code sprintf('%%\t\t\t  into this dimension ordering. If the default is incorrect, \n')];
            code = [code sprintf('%%\t\t\t  use the ''InputDataPermutation'' argument to control the \n')];
            code = [code sprintf('%%\t\t\t  permutation. \n')];
            code = [code sprintf('%%\t\t\t  \n')];
            
            code = [code genCommentedNewline];
            code = [code sprintf('%% PARAMS\t- Network parameters returned by ''importONNXFunction''.\n')];
            code = [code genCommentedNewline];
            code = [code genCommentedNewline];
            
            % Outputs
            code = [code sprintf('%% Outputs:\n%% --------\n')];
            code = [code sprintf('%% %s\n', upper(OutputArgs))];
            code = [code sprintf('%%\t\t\t- Output(s) of the ONNX network.\n')];
            code = [code sprintf('%%\t\t\t  Without permutation, the size(s) of the outputs are:\n')];
            for i=1:numel(graphTranslation.ExternalOutputNames)
                shape = graphTranslation.ExternalOutputShapes{i};
                outputType = graphTranslation.ExternalOutputTypes{i};
                if isempty(shape)
                    shape = {1,1};
                elseif numel(shape)==1
                    shape{2} = 1;
                end
                outputSizes = "[" + strjoin(cellfun(@string, shape), ", ") + "]";
                code = [code sprintf('%%\t\t\t\t  %s:\t\t%s\t\t\t\tType: %s\n', upper(graphTranslation.ExternalOutputNames{i}), outputSizes, outputType)];
            end
            code = [code sprintf('%%\t\t\t  By default, the function will try to permute the output(s) \n')];
            code = [code sprintf('%%\t\t\t  from this dimension ordering into a conventional MATLAB \n')];
            code = [code sprintf('%%\t\t\t  ordering. If the default is incorrect, use the \n')];
            code = [code sprintf('%%\t\t\t  ''OutputDataPermutation'' argument to control the permutation.\n')];
            
            code = [code genCommentedNewline];
            code = [code sprintf('%% STATE\t\t- (Optional) State variables. When TRAINING is true, these will \n')];
            code = [code sprintf('%% \t\t\t  have been updated from the original values in PARAMS.State.\n')];
            code = [code genCommentedNewline];
            code = [code genCommentedNewline];
            
            code = [code sprintf('%%  See also importONNXFunction\n')];
            code = [code newline];
            
        end

        function issues = getTranslationIssues(this)
            issues = this.ToplevelGraphTranslation.TranslationIssues;
        end
    end
end

function code = genFunctionHeaderLine(graphTranslation, outputFcnName)
% Write the function header line. 
    InputArgs = [graphTranslation.ExternalInputNames, {'params', 'varargin'}];
    OutputArgs = [graphTranslation.ExternalOutputNames, {'state'}];
    code = sprintf('function [%s] = %s(%s)\n',...
            strjoin(OutputArgs, ', '), outputFcnName, strjoin(InputArgs, ', '));
end

function code = genCommentedNewline()
    code = sprintf('%% \n');
end

function code = genCallInputValidation(graphTranslation)
    InputArgs = strjoin(graphTranslation.ExternalInputNames, ', ');
    nDataInputs = numel(graphTranslation.ExternalInputNames);
    nDataOutputs = numel(graphTranslation.ExternalOutputNames);
    code = sprintf('%% Parse input arguments\n');
    code = [code sprintf('[inputDataPerms, outputDataPerms, Training] = parseInputs(%s, %d, params, varargin{:});\n', InputArgs, nDataOutputs)];
end

function code = genPermuteInputs(graphTranslation, varargin)
    % Generate code that will permute the input(s) when the user passes
    % permutation vectors to the model function via the
    % 'InputDataPermutation' NVP
    code = sprintf('%% Permute inputs if requested:\n');
    inputNames = graphTranslation.ExternalInputNames;
    inputShapes = graphTranslation.ExternalInputShapes;
    for i=1:numel(inputNames)
        code = [code sprintf('%s = permuteInputVar(%s, inputDataPerms{%d}, %d);\n', ...
            inputNames{i}, inputNames{i}, i, numel(inputShapes{i}))];
    end
end


function code = genPermuteOutputs(graphTranslation, varargin)
    % Generate code that will permute the output(s) when the user passes
    % permutation vectors to the model function via the
    % 'OutputDataPermutation' NVP
    code = sprintf('%% Permute outputs if requested:\n');
    outputNames = graphTranslation.ExternalOutputNames;
    outputShapes = graphTranslation.ExternalOutputShapes;
    
    for i=1:numel(outputNames)
        % To handle custom layers, do not permute outputs that have an empty output shape.
        if ~isempty(outputShapes{i})
            code = [code sprintf('%s = permuteOutputVar(%s, outputDataPerms{%d}, %d);\n', ...
                outputNames{i}, outputNames{i}, i, numel(outputShapes{i}))];
        end
    end
end

function code = genCallPreprocessInput(graphTranslation)
    code = sprintf('%% Preprocess the input data and arguments:\n');
    inputNames = graphTranslation.ExternalInputNames;
    inputNamesString = strjoin(inputNames, ', ');
    code = [code sprintf('[%s, Training, outputDataPerms, anyDlarrayInputs] = preprocessInput(%s, params, varargin{:});\n', inputNamesString, inputNamesString)];
end

function code = genCallPostprocessOutput(graphTranslation)
    code = sprintf('%% Postprocess the output data\n');
    outputNames = graphTranslation.ExternalOutputNames;
    outputNamesString = strjoin(outputNames, ', ');
    code = [code sprintf('[%s] = postprocessOutput(%s, outputDataPerms, anyDlarrayInputs, Training, varargin{:});\n', outputNamesString, outputNamesString)];
end

function code = genPreprocessInputFunction(graphTranslation, isGeneratingCustomLayer)
    inputNames = graphTranslation.ExternalInputNames;
    inputNamesString = strjoin(inputNames, ', ');
    code = sprintf('function [%s, Training, outputDataPerms, anyDlarrayInputs] = preprocessInput(%s, params, varargin)\n', inputNamesString, inputNamesString);
    code = [code genCallInputValidation(graphTranslation)];                     % Call function to validate inputs
    code = [code genAnyDlarray(graphTranslation)];
    code = [code genMakeInputsDlarrays(graphTranslation)];
    code = [code genPermuteInputs(graphTranslation)];
    if ~isGeneratingCustomLayer
        code = [code genCheckInputSizes(graphTranslation)];
    end
    code = [code sprintf('end\n\n')];
end

function code = genPostprocessOutputFunction(graphTranslation)
    outputNames = graphTranslation.ExternalOutputNames;
    outputNamesString = strjoin(outputNames, ', ');
    code = sprintf('function [%s] = postprocessOutput(%s, outputDataPerms, anyDlarrayInputs, Training, varargin)\n', outputNamesString, outputNamesString);
    code = [code genSetOutputTypes(graphTranslation)];
    code = [code genPermuteOutputs(graphTranslation)];
    code = [code sprintf('end\n\n')];
end

function code = genAnyDlarray(graphTranslation)
    inputNameList = strjoin(graphTranslation.ExternalInputNames, ', ');
    code = sprintf('anyDlarrayInputs = any(cellfun(@(x)isa(x, ''dlarray''), {%s}));\n', inputNameList);
end

function code = genMakeInputsDlarrays(graphTranslation)
    code = sprintf('%% Make the input variables into unlabelled dlarrays:\n');
    for i = 1:numel(graphTranslation.ExternalInputNames)
        name = graphTranslation.ExternalInputNames{i};
        code = [code sprintf('%s = makeUnlabeledDlarray(%s);\n', name, name)];
    end
end

function code = genPackageVariables(graphTranslation, isGeneratingCustomLayer)
    if isGeneratingCustomLayer
        % When isGeneratingCustomLayer=true, get the input tensor ranks
        % (numdims) from the runtime numDims vars.
        for i=1:numel(graphTranslation.ExternalInputNames)
            tensorName = graphTranslation.ExternalInputNames{i};
            numDimsName = string(tensorName) + "NumDims";
            ExternalInputRanks(i) = numDimsName; 
        end
    else
        for i = 1:numel(fieldnames(graphTranslation.ExternalInputRanks))
            name = graphTranslation.ExternalInputNames{i};
            inputRank = string(graphTranslation.ExternalInputRanks.(name));
            if ~isempty(inputRank)
                ExternalInputRanks(i) = inputRank;
            else
                % Assume empty input rank implies a scalar input
                ExternalInputRanks(i) = "0";
            end
        end    
    end
    code = sprintf('%% Put all variables into a single struct to implement dynamic scoping:\n');
    code = [code ...
            sprintf('[Vars, NumDims] = packageVariables(params, {%s}, {%s}, [%s]);\n',...
            strjoin(quoteCellstr(graphTranslation.ExternalInputNames), ', '), ...
            strjoin(graphTranslation.ExternalInputNames, ', '),...
            strjoin(ExternalInputRanks))];
end

function S = quoteCellstr(C)
    S = string(C);
    S = arrayfun(@(s)strjoin(["'" s "'"], ""), S);
end

function code = genCallTopLevelGraph(graphTranslation)
    InputRankStrings = cellfun(@(v)sprintf('NumDims.%s', v), graphTranslation.ExternalInputNames, 'UniformOutput',false);
    AllInputStrings = [graphTranslation.ExternalInputNames, InputRankStrings {'Vars', 'NumDims', 'Training', 'params.State'}];
    OutputRankStrings = cellfun(@(v)sprintf('%sNumDims', v), graphTranslation.ExternalOutputNames, 'UniformOutput', false);
    AllOutputStrings = [graphTranslation.ExternalOutputNames, OutputRankStrings, {'state'}];
    code = sprintf('%% Call the top-level graph function:\n');
    code = [code sprintf('[%s] = %s(%s);\n', strjoin(AllOutputStrings, ', '), graphTranslation.GraphName, strjoin(AllInputStrings, ', '))];
    % Note: No need to add a line to assign State into Vars because we're done
    % using State at this point.
end

function code = genSetOutputTypes(graphTranslation)
    code = sprintf('%% Set output type:\n');
    code = [code sprintf('if ~anyDlarrayInputs && ~Training\n')];
    for i = 1:numel(graphTranslation.ExternalOutputNames)
        name = graphTranslation.ExternalOutputNames{i};
        code = [code sprintf('if isdlarray(%s)\n%s = extractdata(%s);\nend\n', name, name, name)];
    end
    code = [code sprintf('end\n')];
end

function code = genInputValidationFunction(graphTranslation, outputFcnName, varargin)
    InputArgs = strjoin(graphTranslation.ExternalInputNames, ', ');
    % OutputArgs = strjoin(graphTranslation.ExternalOutputNames, ', ');
    code = sprintf('function [inputDataPerms, outputDataPerms, Training] = parseInputs(%s, numDataOutputs, params, varargin)\n', InputArgs);
    code = [code sprintf('%% Function to validate inputs to %s:\n', outputFcnName)];
    % code = [code sprintf('import nnet.internal.cnn.onnx.*\n')];
    code = [code sprintf('p = inputParser;\n')];
    code = [code sprintf('isValidArrayInput = @(x)isnumeric(x) || isstring(x);\n')];
    code = [code sprintf('isValidONNXParameters = @(x)isa(x, ''ONNXParameters'');\n')];
    for i=1:numel(graphTranslation.ExternalInputNames)
        inputName = graphTranslation.ExternalInputNames{i};
        code = [code sprintf('addRequired(p, ''%s'', isValidArrayInput);\n', inputName)];
    end
    code = [code sprintf('addRequired(p, ''params'', isValidONNXParameters);\n')];
    code = [code sprintf('addParameter(p, ''InputDataPermutation'', ''auto'');\n')];
    code = [code sprintf('addParameter(p, ''OutputDataPermutation'', ''auto'');\n')];
    code = [code sprintf('addParameter(p, ''Training'', false);\n')];
    code = [code sprintf('parse(p, %s, params, varargin{:});\n', InputArgs)];
    code = [code sprintf('inputDataPerms = p.Results.InputDataPermutation;\n')];
    code = [code sprintf('outputDataPerms = p.Results.OutputDataPermutation;\n')];
    code = [code sprintf('Training = p.Results.Training;\n')];
    code = [code sprintf('if isnumeric(inputDataPerms)\ninputDataPerms = {inputDataPerms};\nend\n')];
    code = [code sprintf('if isstring(inputDataPerms) && isscalar(inputDataPerms) || ischar(inputDataPerms) \ninputDataPerms = repmat({inputDataPerms},1,%d);\nend\n', numel(graphTranslation.ExternalInputNames))];
    code = [code sprintf('if isnumeric(outputDataPerms)\noutputDataPerms = {outputDataPerms};\nend\n')];
    code = [code sprintf('if isstring(outputDataPerms) && isscalar(outputDataPerms) || ischar(outputDataPerms) \noutputDataPerms = repmat({outputDataPerms},1,numDataOutputs);\nend\n')];
    code = [code sprintf('end\n\n')];
end

function code = genCheckInputSizes(graphTranslation, varargin)
    % Generate code that will check whether the input sizes match those
    % requried by the ONNX model
    code = sprintf('%% Check input size(s):\n');
    inputNames = graphTranslation.ExternalInputNames;
    inputShapes = graphTranslation.ExternalInputShapes;
    for i=1:numel(inputShapes)
        inputShapeStr = cell2string(inputShapes{i});
        code = [code sprintf('checkInputSize(size(%s), %s, "%s");\n', inputNames{i}, inputShapeStr, inputNames{i})]; %#ok<AGROW>
    end
end

function s = cell2string(c)
    s = "";
    for i=numel(c):-1:1
        if isnumeric(c{i})
            s(i) = sprintf("%d", c{i});
        else
            s(i) = sprintf("'%s'", c{i});
        end
    end
    s = sprintf("{%s}", join(s));
end

function code = genSpkgUtilityFunctions()
    code = '';
    code = [code sprintf('\n%%%% Utility functions:\n')];
    % Alphabetical order
    UtilityFcns = ["appendStructs", "checkInputSize", "makeUnlabeledDlarray",...
        "packageVariables", "permuteInputVar", "permuteOutputVar", "updateStruct"];
    for i = 1:numel(UtilityFcns)
        fullPath = which("nnet.internal.cnn.onnx.fcn." + UtilityFcns(i));
        fileID = fopen(fullPath);
        if fileID == -1
            error(message('nnet_cnn_onnx:onnx:FileOpen', fullPath));
        end
        A = fread(fileID,'*char')';
        fclose(fileID);
        code = [code newline A]; %#ok<AGROW>
    end
end

function code = genIncludedFunctionDefinitions(IncludedFcns)  
    code = '';
    if ~isempty(IncludedFcns)
        code = [code sprintf('\n%%%% dlarray functions implementing ONNX operators:\n')];
    end
    for i = 1:numel(IncludedFcns)
        fullPath = which("nnet.internal.cnn.onnx.fcn." + IncludedFcns(i));
        fileID = fopen(fullPath);
        if fileID == -1
            error(message('nnet_cnn_onnx:onnx:FileOpen', IncludedFcns(i)));
        end
        opFcnStr = fread(fileID,'*char')';        
        fclose(fileID);
        code = [code newline opFcnStr]; %#ok<AGROW>
    end
end