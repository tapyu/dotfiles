classdef TranslatorForGRU < nnet.internal.cnn.onnx.OperatorTranslator
    % A class to translate ONNX GRU operators into MATLAB layers

    % Copyright 2021-2024 The MathWorks, Inc.

    properties(SetAccess = protected)
        % Operator attributes
        activation_alpha
        activation_beta
        activations
        clip
        direction
        hidden_size
        linear_before_reset
        layout

        % Other properties
        LayerName
    end

    methods
        % Do initial setup. this.Node has been set already.
        function initialize(this)
            this.IsSeedOperator = true;
            this.CanOverwriteTensorLabels = true;

            % Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
            % To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
            AttributeTable = cell2table({
                "activation_alpha"	  "FLOATS"    true    0
                "activation_beta" 	  "FLOATS"    true    0
                "activations"         "STRINGS"   true    []
                "clip"                "FLOAT"     true    []
                "direction"           "STRING"    true    "forward"
                "hidden_size"    	  "INT"       false   []
                "linear_before_reset" "INT"       true    0
                "layout"              "INT"       true    0
                });
            % Parse the attributes
            [this.activation_alpha, this.activation_beta, this.activations, this.clip, this.direction, this.hidden_size, this.linear_before_reset, this.layout] = ...
                nnet.internal.cnn.onnx.parseNodeAttributes(this.Node, AttributeTable);

            % Set layer name
            this.LayerName = this.Node.name;
        end

        % Given the node and the current input and output formats, update
        % the formats in the specified direction to the extent possible.
        % Direction is either "forward" or "reverse". Each format variable
        % is a string array, where each string is one of the members of the
        % constant 'SupportedONNXLabels' property, or empty. Set a format
        % string to empty if its format cannot be determined.
        function [inputTensorFormats, outputTensorFormats] = propagateTensorFormats(this, direction, inputTensorFormats, outputTensorFormats)
            % GRU is a seed operator, so just set all input and output
            % formats.
            if this.layout==0
                % layout==0:
                % X.shape = [seq_length, batch_size, input_size]
                % Y.shape = [seq_length, num_directions, batch_size, hidden_size]
                inputTensorFormats(1) = "TBC";
                if numel(this.Node.output)>=1 && ~isempty(this.Node.output{1}) && this.direction=="forward"
                    % First output is [seq_length, num_directions, batch_size, hidden_size]
                    outputTensorFormats(1) =  "T1BC";
                end
                % layout 1 is not currently supported in any third party libraries, so the following code is disabled.
                % Do not propagate format.
                %             else
                %                 % layout==1:
                %                 % X.shape = [batch_size, seq_length, input_size]
                %                 % Y.shape = [batch_size, seq_length, num_directions, hidden_size]
                %                 inputTensorFormats(1) = "BTC";
                %                 if numel(this.Node.output)>=1 && ~isempty(this.Node.output{1}) && this.direction=="forward"
                %                     % First output is [batch_size, seq_length, num_directions, hidden_size]
                %                     outputTensorFormats(1) =  "BT1C";
                %                 end
            end

            % Under both layouts:
            % initial_h.shape = Y_h.shape = initial_c.shape = Y_c.shape = [num_directions, batch_size, hidden_size]
            if numel(this.Node.input) >= 6 && ~isempty(this.Node.input{6}) && this.direction=="forward"
                % Sixth input, initial_h, is [num_directions, batch_size, hidden_size]
                inputTensorFormats(6) = "1BC";
            end
            if numel(this.Node.output)>=2 && ~isempty(this.Node.output{2}) && this.direction=="forward"
                % Second output, Y_h, is [num_directions, batch_size, hidden_size]
                outputTensorFormats(2) = "1BC";
            end
        end

        % Translate the node into a MATLAB layer, and return any
        % translation issues.
        function [Layer, issues] = translateIntoLayer(this, inputTensorFormats, outputTensorFormats)
            Layer = [];
            issues = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
            if validLabels(this, inputTensorFormats, outputTensorFormats)
                if isempty(this.activations)
                    zrAct  = 'sigmoid';
                    hidAct = 'tanh';
                else
                    zrAct  = iTranslateActivationFunction(this.activations{1});
                    hidAct = iTranslateActivationFunction(this.activations{2});
                end

                % Check that activation functions are valid
                if ~ismember(zrAct, {'sigmoid', 'hard-sigmoid'})
                    issues = [issues nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                        message('nnet_cnn_onnx:onnx:UnsupportedGateActivation'))];
                    Layer = [];
                    return;
                end
                if ~ismember(hidAct, {'tanh', 'softsign', 'relu'})
                    issues = [issues nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                        message('nnet_cnn_onnx:onnx:UnsupportedHiddenActivation'))];
                    Layer = [];
                    return;
                end

                % Determine HasStateInputs, and get initial_h if it's an initializer:
                HasStateInputs = false;
                H = zeros(this.hidden_size,1,'single');
                if numel(this.Node.input) >= 6 && ~isempty(this.Node.input{6})
                    % An initial_h name is there
                    hName = this.Node.input{6};
                    if isTensorInitializer(this.GraphProtoManager, hName)
                        % initial_h is an initializer. Get it.
                        H = single(initializerRawData(this.GraphProtoManager, hName));    % ONNX shape is [num_directions, batch_size, hidden_size], row-major.
                        [H,HShapeIssue] = iFixInitialhShape(this, H);
                        if ~isempty(HShapeIssue)
                            issues = [issues, HShapeIssue];
                            Layer = [];
                            return;
                        end
                    else
                        % initial_h is an activation tensor.
                        HasStateInputs = true;
                    end
                end

                % Determine OutputMode and HasStateOutputs:
                outputUsed = outputUsedInNetwork(this, [1,2]);
                hasOut1 = numel(this.Node.output)>=1 && ~isempty(this.Node.output{1}) && outputUsed(1);
                hasOut2 = numel(this.Node.output)>=2 && ~isempty(this.Node.output{2}) && outputUsed(2);
                if hasOut1
                    % Output Y is present.
                    OutputMode = 'sequence';
                else
                    % Output Y is absent, so Y_h must be the output.
                    OutputMode = 'last';
                end
                if strcmp(this.TargetNetwork,"dagnetwork")
                    % Both Y and Y_h have to be present for DAGNetwork
                    % Otherwise we will get unconnected output error during
                    % assembly
                    HasStateOutputs = hasOut1 && hasOut2; 
                else
                    % dlnetwork
                    % Y_h is present, hence state outputs maybe expected
                    HasStateOutputs = hasOut2;            
                end

                if numel(this.Node.input) >= 4 && ~isempty(this.Node.input{4})
                    % biases: optional in ONNX - we create biases before other weights
                    % because the size of bias determines the value to be given to
                    % resetGateMode, which is needed for constructing gru layer.
                    biasName = this.Node.input{4};
                    B = single(initializerRawData(this.GraphProtoManager, biasName)); % ONNX shape is [num_directions, 6*hidden_size], row-major.
                else
                    B = zeros(3*this.hidden_size,1,'single');
                end

                if this.linear_before_reset == 0
                    resetGateMode = 'before-multiplication';
                elseif length(B) > 3*this.hidden_size && any(B(3*this.hidden_size+1:end),'all')
                    resetGateMode = 'recurrent-bias-after-multiplication';
                else
                    resetGateMode = 'after-multiplication';
                end

                % Create DLT layer
                if strcmp(this.direction, 'forward')
                    [Layer, constructionIssue] = constructLayer(this, 'gruLayer', this.LayerName, this.Node, ...
                        this.hidden_size, 'Name', this.LayerName, ...
                        'OutputMode', OutputMode,...
                        'HasStateInputs', HasStateInputs,...
                        'HasStateOutputs', HasStateOutputs,...
                        'StateActivationFunction', hidAct,...
                        'GateActivationFunction', zrAct, ...
                        'ResetGateMode', resetGateMode);
                    issues = [issues constructionIssue];
                else
                    issues = [issues nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                        message('nnet_cnn_onnx:onnx:UnsupportedGRUDirection'))];
                    Layer = [];
                    return;
                end

                inputWeightName = this.Node.input{2};
                recurrentWeightName = this.Node.input{3};

                inputWeightIssue = iVerifyInitializer(this, inputWeightName);
                recurrentWeightIssue = iVerifyInitializer(this, recurrentWeightName);

                if ~isempty(inputWeightIssue) || ~isempty(recurrentWeightIssue)
                    issues = [issues, inputWeightIssue, recurrentWeightIssue];
                    Layer = [];
                    return;
                end

                % input weights
                W = single(initializerRawData(this.GraphProtoManager, inputWeightName));     % ONNX shape is [num_directions, 3*hidden_size, input_size], row-major.
                W = iFixWeightsShape(this, W);
                Layer.InputWeights = W;

                % recurrent weights
                R = single(initializerRawData(this.GraphProtoManager, recurrentWeightName)); % ONNX shape is [num_directions, 3*hidden_size, hidden_size], row-major.
                R = iFixWeightsShape(this, R);

                Layer.RecurrentWeights = R;
                Layer.InputWeights = W;

                % initial_b: optional in ONNX
                if numel(this.Node.input) >= 4 && ~isempty(this.Node.input{4})
                    B = iFixBiasShape(this, B, resetGateMode);
                end
                Layer.Bias = B;

                % Assign initial H into the layer:
                if ~HasStateInputs
                    Layer.HiddenState = H;
                end
            end
        end
    end

    methods(Access=protected)
        function tf = validLabels(this, inputTensorFormats, outputTensorFormats)
            % First input must be 'TBC'
            % First output (if given) must be 'T1BC'
            % Second output (if given) must be '1BC'
            outputUsed = outputUsedInNetwork(this, [1,2]);
            hasOut1 = numel(this.Node.output)>0 && ~isempty(this.Node.output{1}) && outputUsed(1);
            hasOut2 = numel(this.Node.output)>1 && ~isempty(this.Node.output{2}) && outputUsed(2);
            tf = inputTensorFormats(1)=="TBC" && ...
                (~hasOut1 || outputTensorFormats(1)=="T1BC") && ...
                (~hasOut2 || outputTensorFormats(2)=="1BC");
        end

        function issue = iVerifyInitializer(this, key)
            issue = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
            if ~isTensorInitializer(this.GraphProtoManager, key)
                issue =  nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                    message('nnet_cnn_onnx:onnx:LSTMFromOperator'));
            end
        end

        function [H,issue] = iFixInitialhShape(this, H)
            % ONNX shape is [num_directions, batch_size, hidden_size], row-major. The
            % only allowed direction is 'forward', hence in the case dimensions don't
            % match, the ONNX batch_size is higher than 1, which is not supported in
            % DLT.
            issue = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
            if numel(H) == this.hidden_size
                % MATLAB order is [hidden_size,1] if forward, hence H is column-major.
                H = reshape(H, [this.hidden_size, 1, 1]);
            else
                issue = nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                    message('nnet_cnn_onnx:onnx:UnsupportedGRUBatchedState'));
            end
        end

        function B = iFixBiasShape(this, B, resetGateMode)
            % ONNX order is ZRH
            % MATLAB order is  RZH
            zrhRows = reshape(1:3*this.hidden_size, [], 3);             % Columns of this correspond to ONNX's zrc ordering.
            rzhRows = zrhRows(:, [2 1 3]);                              % Columns of this correspond to DLT's rzh ordering.
            rzhInd  = rzhRows(:);                                       % A column vector containing DLT's rzh ordering.
            switch resetGateMode
                case 'before-multiplication'
                    % B is [1, 6*hidden_size], row-major. (Same col-major)
                    B = B(rzhInd)+B(rzhInd+3*this.hidden_size);  % First half of B is now in DLT's required rzh order.
                case 'after-multiplication'
                    B = B(rzhInd);
                otherwise
                    B(1:6*this.hidden_size) = [B(rzhInd), B(rzhInd+3*this.hidden_size)];
            end
            B = B(:);
        end

        function W = iFixWeightsShape(this, W)
            % ONNX order is ZRH
            % MATLAB order is  RZH
            zrhRows = reshape(1:3*this.hidden_size, [], 3);             % Columns of this correspond to ONNX's zrh ordering.
            rzhRows = zrhRows(:, [2 1 3]);                              % Columns of this correspond to DLT's rzh ordering.
            rzhInd  = rzhRows(:);                                       % A column vector containing DLT's rzh ordering.
            inputSize   = numel(W)/(3*this.hidden_size);                % W is [1, 3*hidden_size, input_size], row-major.
            W           = reshape(W, [inputSize, 3*this.hidden_size]);  % W is now [inputSize, 3*hiddenSize] col-major (using the "fliplr theorem").
            W           = W';                                           % W is now [3*hiddenSize, inputSize].
            W           = W(rzhInd, :);                                 % Rows of W are now in DLT's required rzh order.
        end
    end
end


function s = iTranslateActivationFunction(s)
switch s
    case 'Sigmoid'
        s = 'sigmoid';
    case 'Softsign'
        s = 'softsign';
    case 'Tanh'
        s = 'tanh';
    case 'Relu'
        s = 'relu';
    case 'HardSigmoid'
        s = 'hard-sigmoid';
    otherwise
        % Maintain name for error message
end
end
