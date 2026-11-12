classdef TranslatorForGemm < nnet.internal.cnn.onnx.OperatorTranslator
    % A class to translate ONNX Gemm operators into MATLAB layers
    
    % Copyright 2021-2024 The MathWorks, Inc.
    
    properties(SetAccess = protected)
        % Operator Attributes
        alpha
        beta
        broadcast
        transA
        transB
    end
    
    methods
        % Do initial setup. this.Node has been set already.
        function initialize(this)
            % Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
            % To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
            AttributeTable = cell2table({
                "alpha"        "FLOAT"       true    1
                "beta"         "FLOAT"       true    1
                "broadcast"      "INT"       true    0
                "transA"         "INT"       true    0
                "transB"         "INT"       true    0
                });
            % Parse the attributes
            [this.alpha, this.beta, this.broadcast, this.transA, this.transB] = nnet.internal.cnn.onnx.parseNodeAttributes(this.Node, AttributeTable);
        end
        
        % Given the node and the current input and output formats, update
        % the formats in the specified direction to the extent possible.
        % Direction is either "forward" or "reverse". Each format variable
        % is a string array, where each string is one of the members of the
        % constant 'SupportedONNXLabels' property, or empty. Set a format
        % string to empty if its format cannot be determined.
        function [inputFormats, outputFormats] = propagateTensorFormats(this, direction, inputFormats, outputFormats)
            % Y = alpha * A' * B' + beta * C, where the transposes are
            % optional and indicated by transA and transB. The only case of
            % Gemm that we propagate and support is when A's format is
            % "BC", A is not transposed, and B and C are initializers, in
            % which case it can be imported into a fullyConnectedLayer.
            if ~this.transA && isNodeInputInitializer(this.GraphProtoManager, this.Node, 2) && ...
                    (numel(this.Node.input) < 3 || isempty(this.Node.input{3})) || isNodeInputInitializer(this.GraphProtoManager, this.Node, 3)
                if direction=="forward"
                    if inputFormats(1)=="BC"
                        outputFormats(1) = "BC";
                    end
                else
                    % backward
                    if outputFormats(1)=="BC"
                        inputFormats(1) = "BC";
                    end
                end
            end
        end
        
        % Translate the node into a MATLAB layer, and return any
        % translation issues.
        function [Layer, issues] = translateIntoLayer(this, inputTensorFormats, outputTensorFormats)
            Layer = [];
            issues = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
            % Y = alpha * A' * B' + beta * C, where the transposes are
            % optional and indicated by transA and transB. The only case of
            % Gemm that we propagate and support is when A and Y's format
            % is "BC", A is not transposed, and B and C are initializers,
            % in which case it can be imported into a fullyConnectedLayer.
            if inputTensorFormats(1)=="BC" && outputTensorFormats=="BC" && ~this.transA && ...
                    isNodeInputInitializer(this.GraphProtoManager, this.Node, 2) && ...
                    (numel(this.Node.input) < 3 || isempty(this.Node.input{3}) || isNodeInputInitializer(this.GraphProtoManager, this.Node, 3))
                % Create parameters
                weightName  = this.Node.input{2};                                                   % Get weight matrix to determine numFilters/numUnits
                weightDim   = initializerSize(this.GraphProtoManager, weightName);                  % Either [X F] or [F X]
                rawWeight   = this.alpha*single(initializerRawData(this.GraphProtoManager, weightName));
                if this.transB
                    numUnits = double(weightDim(1));                        % weights are [F X]
                else
                    numUnits = double(weightDim(2));                        % weights are [X F]
                end
                if numel(this.Node.input)>2 && ~isempty(this.Node.input{3})
                    biasName = this.Node.input{3};                                                  % Bias
                    biasDim = initializerSize(this.GraphProtoManager, biasName);                    % Could be 0D, 1D or 2D. Supported shapes are [], [F] and [1 F]
                    rawBias = this.beta*single(initializerRawData(this.GraphProtoManager, biasName));
                else
                    biasDim = [];
                    rawBias = zeros(numUnits, 1);
                end
                if ~biasDimOK(biasDim)
                    issues = [issues nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                        message('nnet_cnn_onnx:onnx:GemmBiasDim'))];
                    Layer = [];
                    return;
                end
                % Create FullyConnectedLayer
                [Layer, constructionIssue] = constructLayer(this, 'fullyConnectedLayer', this.Node.name, this.Node, numUnits, 'Name', this.Node.name);
                issues = [issues constructionIssue];
                if isempty(Layer)
                    return;
                end
                % Import weights
                if isempty(initializerRawData(this.GraphProtoManager, weightName))
                    issues = [issues nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                        message('nnet_cnn_onnx:onnx:WeightHasNoInitializer'))];
                    Layer = [];
                    return;
                end
                % Transform weights to get colmajor [F X].
                if this.transB
                    % Convert rowmajor [F X] to colmajor [F X]
                    weight = reshape(rawWeight, fliplr(weightDim));         % Now it's colmajor [X F].
                    weight = weight';                                       % Now it's colmajor [F X].
                else
                    % Convert rowmajor [X F] to colmajor [F X]. Just
                    % reshape. The data is already in the right linear
                    % order.
                    weight = reshape(rawWeight, fliplr(weightDim));         % Now it's colmajor [F X].
                end
                Layer.Weights = weight;
                Layer.Bias    = rawBias(:);
            end
        end
    end
end

function tf = biasDimOK(biasDim)
% Could be 0D, 1D or 2D. Supported shapes are [], [F] and [1 F]
tf = numel(biasDim)<=1 || numel(biasDim)==2 && biasDim(1)==1;
end
