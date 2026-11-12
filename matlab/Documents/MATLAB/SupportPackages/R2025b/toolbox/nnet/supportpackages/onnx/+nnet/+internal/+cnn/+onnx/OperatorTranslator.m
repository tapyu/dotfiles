classdef OperatorTranslator < handle & matlab.mixin.Heterogeneous
    % A base class for defining subclasses, each of which knows how to
    % translate a single ONNX operator. Subclasses should be named
    % TranslatorFor<op>. Given an ONNX NodeProto containing the operator,
    % the subclass knows how to: (1) Propagate input and output tensor
    % formats in both directions. (2) Decide whether the node is importable
    % into a MATLAB layer, and (3) Translate the node into a MATLAB layer.
    % Subclasses must define the three abstract methods below.

    %   Copyright 2021-2024 The MathWorks, Inc.    

    % Set these properties to true in the initialize() method of subclasses
    % as needed:
    properties
        IsSeedOperator = false;
        CanOverwriteTensorLabels = false;
        CanPropagateSingletonFormats = false;
    end

    methods(Abstract)
        % Do initial setup. this.Node has been set already.
        initialize(this)

        % Given the node and the current input and output formats, update
        % the formats in the specified direction to the extent possible.
        % Direction is either "forward" or "reverse". Each format variable
        % is a string array, where each string is one of the members of the
        % constant 'SupportedONNXLabels' property, or "". Set a format
        % string to "" if its format cannot be determined.
        [inputFormats, outputFormats] = propagateTensorFormats(this, direction, inputFormats, outputFormats)

        % Translate the node into a MATLAB layer, and return any
        % translation issues.
        [Layer, issues] = translateIntoLayer(this, inputTensorFormats, outputTensorFormats)
    end

    properties(Constant)
        % All supported tensor labels, and their MATLAB interpretation
        SupportedONNXLabels = [...
            "BC" ...                            % feature batch: CB
            "BCSS" "BSSC" "CSS" "SSC" ...       % 2d image batch: SSCB
            "BCSSS" "BSSSC" "CSSS" "SSSC"...    % 3d image batch: SSSCB
            "TBC" "BCT" "BTC"  ...              % vector sequence batch: CBT
            "1BC"  ...                          % vector sequence batch's last time step: CB1=CB
            "T1BC" ...                          % vector sequence batch output from an ONNX LSTM operator: CBT
            "TBCSS" ...                         % 2d image sequence batch: SSCBT
            "TBCSSS" ...                        % 3d image sequence batch: SSSCBT
            ];
    end

    properties(SetAccess=protected)
        % The node to be translated
        Node nnet.internal.cnn.onnx.NodeProto

        % The ONNX graph
        GraphProtoManager nnet.internal.cnn.onnx.GraphProtoManager

        % Whether we're generating custom layers
        GenerateCustomLayers

        % Whether the user has requested a DAG or dlnetwork
        TargetNetwork  
    end

    methods(Static)
        % Factory method for creating the right subclass
        function translatorInstance = create(nodeProto, graphProtoManager, generateCustomLayers, targetNetwork)
            % Create an instance, then call its initialize method
            opName = string(nodeProto.op_type);
            translatorName = "nnet.internal.cnn.onnx.TranslatorFor" + opName;
            if exist(translatorName, 'class')
                translatorInstance = feval(translatorName, nodeProto, graphProtoManager, generateCustomLayers, targetNetwork);
            else
                translatorInstance = nnet.internal.cnn.onnx.TranslatorForFullyUnsupportedLayer(nodeProto, graphProtoManager, generateCustomLayers, targetNetwork);
            end
        end
    end

    methods(Access=protected)
        % Don't call the constructor, call the static factory method
        function this = OperatorTranslator(nodeProto, graphProtoManager, generateCustomLayers, targetNetwork)
            arguments
                nodeProto nnet.internal.cnn.onnx.NodeProto
                graphProtoManager nnet.internal.cnn.onnx.GraphProtoManager
                generateCustomLayers logical
                targetNetwork string = 'dlnetwork'
            end
            this.Node = nodeProto;
            this.GraphProtoManager = graphProtoManager;
            this.GenerateCustomLayers = generateCustomLayers;
            this.TargetNetwork = targetNetwork;
            initialize(this);
        end

        %% Helper methods for tensor label propagation:
        function [inputFormats, outputFormats] = propagateMISOBroadcastOp(this, direction, inputFormats, outputFormats)
            % Any MISO operator that supports implicit expansion of the
            % inputs
            tempInputFormats = inputFormats;
            if direction=="forward"
                % Do temporary input-to-input propagation: Input tensors
                % must agree starting from the right. Propagate to
                % initializers with known rank.
                [maxLen,pos] = max(strlength(inputFormats));
                if maxLen>0
                    longestFmt = inputFormats(pos);
                    tempInputFormats = labelBroadcastInitializers(this, this.Node.input, inputFormats, longestFmt);
                end
                % If all tempInputFormats are known, propagate the longest
                % one to the output
                if all(tempInputFormats~="")
                    [~,loc] = max(strlength(tempInputFormats));
                    outputFormats(1) = tempInputFormats(loc);
                end
            else % backward: 
                % (1) Use the output to temporarily label all input
                % initializers.
                if outputFormats~=""
                    tempInputFormats = labelBroadcastInitializers(this, this.Node.input, inputFormats, outputFormats);
                end
                % (2) If any inputs are network external inputs with
                % numdims equal to that of the outout of this node, set the
                % input format to be the same as the output of this node.
                for i=1:numel(this.Node.input)
                    tensorName = this.Node.input{i};
                    if inputFormats(i)=="" && ismember(tensorName, this.GraphProtoManager.ExternalInputNames)
                        tensorSize = this.GraphProtoManager.onnxInputSizeCell(tensorName);
                        if numel(tensorSize)==strlength(outputFormats(1))
                            inputFormats(i) = outputFormats(1);
                        end
                    end
                end
                % (3) If exactly one input is unknown, and the output is
                % longer than all known inputs, then the unknown input must
                % be the same format as the output
                if sum(tempInputFormats=="")==1 && strlength(outputFormats(1))>max(strlength(tempInputFormats))
                    inputFormats(tempInputFormats=="") = outputFormats(1);
                end
            end
        end

        function [inputFormats, outputFormats] = propagateMISOPassthroughOp(~, direction, inputFormats, outputFormats)
            % A MISO operator where all inputs and the output must have the
            % same label. Example: Concat
            if direction=="forward"
                % Set the output to the first non-"" input
                idx = find(inputFormats ~= "", 1);
                if ~isempty(idx)
                    outputFormats(1) = inputFormats(idx);
                end
            else
                % backward Set all input strings to the output string
                if outputFormats(1) ~= ""
                    inputFormats(:) = outputFormats(1);
                end
            end
        end

        function [inputFormats, outputFormats] = propagateSIMOPassthroughOp(~, direction, inputFormats, outputFormats)
            % A SIMO operator where all outputs and the input must have the
            % same label. Example: Split-2
            if direction=="forward"
                % Set all output strings to the input string
                if inputFormats(1) ~= ""
                    outputFormats(:) = inputFormats(1);
                end
            else
                % backward Set the input to the first non-"" output
                idx = find(outputFormats ~= "", 1);
                if ~isempty(idx)
                    inputFormats(1) = outputFormats(idx);
                end
            end
        end

        function [inputFormats, outputFormats] = propagateSISOPassthroughOp(~, direction, inputFormats, outputFormats)
            % Propagate the label unchanged in the specified direction
            % between the first input and the first output tensors. The
            % operator may be MIMO, but only the first input and output are
            % propagated. Examples: Relu, Conv (propagates only the image
            % tensor format)
            if direction=="forward"
                if inputFormats(1)~=""
                    outputFormats(1) = inputFormats(1);
                end
            else
                % backward
                if outputFormats(1)~=""
                    inputFormats(1) = outputFormats(1);
                end
            end
        end

        %% Helper methods for translating into Layers
        function [Layer, issue] = constructLayer(~, constructorName, LayerName, node, varargin)
            issue = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
            try
                Layer = feval(constructorName, varargin{:});
            catch ME
                issue = nnet.internal.cnn.onnx.NodeTranslationError(node, ...
                    message('nnet_cnn_onnx:onnx:LayerConstructorFailed', LayerName, constructorName, ME.message));
                Layer = [];
            end
        end

        function constData = permuteConstInput(~, constData, constDim, knownFormat)
            % Utility function for Add, Sub, Mul, and Div nodes to permute
            % the ONNX raw data for a constant input into the appropriate
            % MATLAB shape for use in ElementwiseAffineLayer. The MATLAB
            % shape can be any shape that is expandable to the known input
            % tensor shape in MATLAB.
            if isempty(constDim)
                % The constant is a scalar. Pass it through unchanged.
                return;
            end
            if numel(constDim) > strlength(knownFormat)
                assert(false, message('nnet_cnn_onnx:onnx:ConstantDimsAndFormatMismatch'));
            end
            constDim = constDim(:)'; % Make sure constDim is a row vector
            switch knownFormat
                case {"BC" "TBC" "1BC" "T1BC"}
                    % The known input format in ONNX is either a feature
                    % batch or vector sequence batch.
                    switch numel(constDim)
                        case 1
                            % C. Make it a column
                            constData = constData(:);
                        case 2
                            % BC, Make it CB colmaj
                            constData = reshape(constData, fliplr(constDim));              	% BC rowmaj --> CB colmaj.
                        case 3
                            % TBC or 1BC. Make it CBT or CB colmaj
                            constData = reshape(constData, fliplr(constDim));              	% TBC rowmaj --> CBT colmaj, or 1BC rowmaj --> CB colmaj.
                        case 4
                            % T1BC. Make it CBT colmaj
                            constData = reshape(constData, fliplr(constDim));              	% T1BC rowmaj --> CB1T colmaj.
                            constData = permute(constData, [1 2 4 3]);                      % CB1T colmaj --> CBT colmaj.
                    end
% In the future this format may be propagated by models with recurrent operators that use layout 1.
% There are no 3p libraries that currently support this, so it is disabled.
%                 case "BT1C"
%                     % The known input format in ONNX is a vector sequence
%                     % batch. In DLT we want CBT      
%                     switch numel(constDim)
%                         case {1 2}
%                             % C or % 1C. Make it a column
%                             constData = constData(:);                            
%                         case 3
%                             % T1C. Make it CBT colmaj
%                             constData = reshape(constData, fliplr(constDim));              	% T1C rowmaj --> C1T==CBT colmaj                          
%                         case 4
%                             % BT1C. Make it CBT colmaj
%                             constData = reshape(constData, fliplr(constDim));               % BT1C rowmaj --> C1TB colmaj
%                             constData = permute(constData, [1 4 3 2]);                      % C1TB colmaj --> CBT colmaj
%                     end
                case "BCT"
                    % The known input format in ONNX is a Batch Channel Sequence 
                    % In DLT we want CBT
                    switch numel(constDim)
                        case 1
                            % C. Make it a column
                            constData = constData(:);
                        case 2
                            % CT, Make it C1T colmaj
                            constData = reshape(constData, fliplr(constDim));              	% CT rowmaj --> TC colmaj.
                            constData = permute(constData, [2 3 1]);                        % TC colmaj --> C1T colmaj
                        case 3
                            % BCT. Make it CBT colmaj
                            constData = reshape(constData, fliplr(constDim));              	% BCT rowmaj --> TCB colmaj.
                            constData = permute(constData, [2 3 1]);                        % TCB colmaj --> CBT colmaj.
                    end
                case "BTC"
                    % The known input format in ONNX is a Batch Sequence Channel 
                    % In DLT we want CBT
                    switch numel(constDim)
                        case 1
                            % C. Make it a column
                            constData = constData(:);
                        case 2
                            % TC, Make it C1T colmaj
                            constData = reshape(constData, fliplr(constDim));              	% TC rowmaj --> CT colmaj.
                            constData = permute(constData, [1 3 2]);                        % CT colmaj --> C1T colmaj
                        case 3
                            % BTC. Make it CBT colmaj
                            constData = reshape(constData, fliplr(constDim));              	% BTC rowmaj --> CTB colmaj.
                            constData = permute(constData, [1 3 2]);                        % CTB colmaj --> CBT colmaj.
                    end
                case {"BCSS" "CSS" "TBCSS"}
                    % The known input format in ONNX is a 2d image batch,
                    % or 2d image batch sequence. In DLT we want HWCB or
                    % HWCBT.
                    switch numel(constDim)
                        case 1
                            % W. Make it 1W colmaj
                            constData = constData(:)';
                        case 2
                            % HW, Make it HW colmaj
                            constData = reshape(constData, fliplr(constDim));              	% HW rowmaj --> WH colmaj.
                            constData = permute(constData, [2 1]);                          % WH colmaj --> HW colmaj.
                        case 3
                            % CHW, Make it HWC colmaj
                            constData = reshape(constData, fliplr(constDim));              	% CHW rowmaj --> WHC colmaj.
                            constData = permute(constData, [2 1 3]);                       	% WHC colmaj --> HWC colmaj.
                        case 4
                            % BCHW. Make it HWCB colmaj
                            constData = reshape(constData, fliplr(constDim));              	% BCHW rowmaj --> WHCB colmaj.
                            constData = permute(constData, [2 1 3 4]);                      % WHCB colmaj --> HWCB colmaj.
                        case 5
                            % TBCHW. Make it HWCBT colmaj
                            constData = reshape(constData, fliplr(constDim));              	% TBCHW rowmaj --> WHCBT colmaj.
                            constData = permute(constData, [2 1 3 4 5]);                    % WHCBT colmaj --> HWCBT colmaj.
                    end
                case {"BSSC" "SSC"}
                    % The known input format in ONNX is a 2d image batch
                    % (channels last), BHWC. The DLT data format is HWCB.
                    switch numel(constDim)
                        case 1
                            % C. Make it 11C colmaj.
                            constData = reshape(constData, [1 1 constDim]);                 % C rowmaj --> 11C colmaj.
                        case 2
                            % WC, Make it 1WC colmaj
                            constData = reshape(constData, fliplr(constDim));              	% WC rowmaj --> CW colmaj.
                            constData = permute(constData, [3 2 1]);                        % CW colmaj --> 1WC colmaj.
                        case 3
                            % HWC. Make it HWC colmaj.
                            constData = reshape(constData, fliplr(constDim));               % HWC rowmaj --> CWH colmaj.
                            constData = permute(constData, [3 2 1]);                        % CWH colmaj --> HWC colmaj.
                        case 4
                            % BHWC. Make it HWCB colmaj
                            constData = reshape(constData, fliplr(constDim));              	% BHWC rowmaj --> CWHB colmaj.
                            constData = permute(constData, [3 2 1 4]);                      % CWHB colmaj --> HWCB colmaj.
                    end
                case {"BCSSS" "CSSS" "TBCSSS"}
                    % The known input format in ONNX is a 3d image batch,
                    % or 3d image batch sequence. In DLT we want HWDCB or
                    % HWDCBT.
                    switch numel(constDim)
                        case 1
                            % D. Make it 11D colmaj
                            constData = reshape(constData, [1 1 constDim]);
                        case 2
                            % WD, Make it 1WD colmaj
                            constData = reshape(constData, fliplr(constDim));              	% WD rowmaj --> DW colmaj.
                            constData = permute(constData, [3 2 1]);                        % DW colmaj --> 1WD colmaj.
                        case 3
                            % HWD, Make it HWD colmaj
                            constData = reshape(constData, fliplr(constDim));              	% HWD rowmaj --> DWH colmaj.
                            constData = permute(constData, [3 2 1]);                       	% DWH colmaj --> HWD colmaj.
                        case 4
                            % CHWD. Make it HWDC colmaj
                            constData = reshape(constData, fliplr(constDim));              	% CHWD rowmaj --> DWHC colmaj.
                            constData = permute(constData, [3 2 1 4]);                      % DWHC colmaj --> HWDC colmaj.
                        case 5
                            % BCHWD. Make it HWDCB colmaj
                            constData = reshape(constData, fliplr(constDim));              	% BCHWD rowmaj --> DWHCB colmaj.
                            constData = permute(constData, [3 2 1 4 5]);                    % DWHCB colmaj --> HWDCB colmaj.
                        case 6
                            % TBCHWD. Make it HWDCBT colmaj
                            constData = reshape(constData, fliplr(constDim));              	% TBCHWD rowmaj --> DWHCBT colmaj.
                            constData = permute(constData, [3 2 1 4 5 6]);                  % DWHCBT colmaj --> HWDCBT colmaj.
                    end
                case {"BSSSC" "SSSC"}
                    % The known input format in ONNX is a 3d image batch
                    % (channels last), BHWDC. The DLT data format is HWDCB.
                    switch numel(constDim)
                        case 1
                            % C. Make it 111C colmaj.
                            constData = reshape(constData, [1 1 1 constDim]);               % C rowmaj --> 111C colmaj.
                        case 2
                            % DC, Make it 11DC colmaj
                            constData = reshape(constData, fliplr(constDim));              	% DC rowmaj --> CD colmaj.
                            constData = permute(constData, [4 3 2 1]);                      % CD colmaj --> 11DC colmaj.
                        case 3
                            % WDC. Make it 1WDC colmaj.
                            constData = reshape(constData, fliplr(constDim));               % WDC rowmaj --> CDW colmaj.
                            constData = permute(constData, [4 3 2 1]);                      % CDW colmaj --> 1WDC colmaj.
                        case 4
                            % HWDC. Make it HWDC colmaj
                            constData = reshape(constData, fliplr(constDim));              	% HWDC rowmaj --> CDWH colmaj.
                            constData = permute(constData, [4 3 2 1]);                      % CDWH colmaj --> HWDC colmaj.
                        case 5
                            % BHWDC. Make it HWDCB colmaj
                            constData = reshape(constData, fliplr(constDim));              	% BHWDC rowmaj --> CDWHB colmaj.
                            constData = permute(constData, [4 3 2 1 5]);                    % CDWHB colmaj --> HWDCB colmaj.
                    end
                otherwise
                    assert(false, message('nnet_cnn_onnx:onnx:UnknownONNXFormat', knownFormat));
            end
        end

        function tf = outputUsedInNetwork(this, outputNums)
            graphNodeInputs = string(allNodeInputNames(this.GraphProtoManager));
            graphOutputs = string(this.GraphProtoManager.OutputNames);
            tf = false(1,numel(outputNums));
            for i = 1:numel(outputNums)
                outputIdx = outputNums(i);
                if numel(this.Node.output) >= outputIdx && ~isempty(this.Node.output{outputIdx}) % Check that the output is set and that its value is not ''.
                    outputName = string(this.Node.output{outputIdx});
                    % Check for references in the top-level graph:
                    if ismember(outputName, [graphNodeInputs graphOutputs])
                        tf(i) = true;
                        continue;
                    end
    
                    % Loop over all nodes, checking for dynamic usage:
                    Nodes = this.GraphProtoManager.Nodes;
                    for j=1:numel(Nodes)
                        if ~isempty(dynamicallyReferencedTensorNamesInNode(this.GraphProtoManager, Nodes(j), outputName))
                            tf(i) = true;
                            continue;
                        end
                    end  
                end
            end
        end
    end

    methods(Access=private)
        function formats = labelBroadcastInitializers(this, tensorNames, formats, aKnownFormat)
            % When tensors are combined using broadcasting, their formats
            % are right-justified. Given a known format of one of the
            % tensors, use right-justification to label all initializers
            % whose rank is less than or equal to the length of that
            % format.
            for i=1:numel(tensorNames)
                if formats(i)==""
                    tensorName = tensorNames{i};
                    if isTensorInitializer(this.GraphProtoManager, tensorName)
                        rank = numel(initializerSize(this.GraphProtoManager, tensorName));
                        if rank==0
                            formats(i) = "0";       % Indicates a zero-dimensional scalar.
                        elseif rank <= strlength(aKnownFormat)
                            formats(i) = extractAfter(aKnownFormat, strlength(aKnownFormat)-rank);
                        end
                    end
                end
            end
        end
    end
end

