classdef TranslatorForLSTM < nnet.internal.cnn.onnx.OperatorTranslator
    % A class to translate ONNX LSTM operators into MATLAB layers

    % Copyright 2021-2024 The MathWorks, Inc.

    properties(SetAccess = protected)
        % Operator attributes
        activation_alpha
        activation_beta
        activations
        clip
        direction
        hidden_size
        input_forget
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
                "activation_alpha"	"FLOATS"    true    0
                "activation_beta" 	"FLOATS"    true    0
                "activations"       "STRINGS"   true    []
                "clip"              "FLOAT"     true    []
                "direction"         "STRING"    true    "forward"
                "hidden_size"    	"INT"       false   []
                "input_forget"    	"INT"       true    0
                "layout"            "INT"       true    0
                });
            % Parse the attributes
            [this.activation_alpha, this.activation_beta, this.activations, this.clip, this.direction, this.hidden_size, this.input_forget, this.layout] = ...
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
            % LSTM is a seed operator, so just set all input and output
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
            if numel(this.Node.input) >= 7 && ~isempty(this.Node.input{7}) && this.direction=="forward"
                % Seventh input, initial_c, is [num_directions, batch_size, hidden_size]                
                inputTensorFormats(7) = "1BC";
            end
            if numel(this.Node.output)>=2 && ~isempty(this.Node.output{2}) && this.direction=="forward"
                % Second output, Y_h, is [num_directions, batch_size, hidden_size]
                outputTensorFormats(2) = "1BC";
            end
            if numel(this.Node.output)>=3 && ~isempty(this.Node.output{3}) && this.direction=="forward"
                % Third output, Y_c, is [num_directions, batch_size, hidden_size]
                outputTensorFormats(3) = "1BC";
            end 
        end     

        % Translate the node into a MATLAB layer, and return any
        % translation issues.
        function [Layer, issues] = translateIntoLayer(this, inputTensorFormats, outputTensorFormats)
            Layer = [];
            issues = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;

            % Check direction
            if ~isequal(this.direction, 'forward')
                issues = nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                    message('nnet_cnn_onnx:onnx:UnsupportedLSTMDirection', this.direction));
                return
            end

            % Check for supported tensor labels
            if ~validLabels(this, inputTensorFormats, outputTensorFormats)
                return
            end

            % Process activations
            if isempty(this.activations)
                this.activations = {'sigmoid', 'tanh', 'tanh'};
            end
            iofAct = iTranslateActivationFunction(this.activations{1});
            cellAct = iTranslateActivationFunction(this.activations{2});
            hidAct = iTranslateActivationFunction(this.activations{3});
            % Check that activation functions are valid
            if ~ismember(iofAct, {'sigmoid', 'hard-sigmoid'})
                issues = [issues nnet.internal.cnn.onnx.NodeTranslationError(this.Node, ...
                    message('nnet_cnn_onnx:onnx:UnsupportedGateActivation'))];
                Layer = [];
                return;
            end
            if ~ismember(cellAct, {'tanh', 'softsign', 'relu'})
                issues = [issues nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                    message('nnet_cnn_onnx:onnx:UnsupportedCellActivation'))];
                Layer = [];
                return;
            end
            if ~ismember(hidAct, {'tanh', 'softsign', 'relu'})
                issues = [issues nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                    message('nnet_cnn_onnx:onnx:UnsupportedHiddenActivation'))];
                Layer = [];
                return;
            end
            if ~isequal(hidAct, cellAct)
                issues = [issues nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                    message('nnet_cnn_onnx:onnx:CellAndHiddenActsUnequal'))];
                Layer = [];
                return;
            end

            % Check that weights and biases are initializers
            inputWeightName = this.Node.input{2};
            inputWeightIssue = iVerifyInitializer(this, inputWeightName);
            recurrentWeightName = this.Node.input{3};
            recurrentWeightIssue = iVerifyInitializer(this, recurrentWeightName);
            biasIssue = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
            if numel(this.Node.input) >= 4 && ~isempty(this.Node.input{4})
                % a bias name is present
                biasName = this.Node.input{4};
                biasIssue = iVerifyInitializer(this, biasName);
            end
            % Exit early if weight or bias issues
            if ~isempty(inputWeightIssue) || ~isempty(recurrentWeightIssue) || ~isempty(biasIssue)
                issues = [issues, inputWeightIssue, recurrentWeightIssue, biasIssue];
                Layer = [];
                return;
            end

            % Get weights
            W = single(initializerRawData(this.GraphProtoManager, inputWeightName));     % ONNX shape is [num_directions, 4*hidden_size, input_size], row-major.
            W = iFixWeightsShape(this, W);
            R = single(initializerRawData(this.GraphProtoManager, recurrentWeightName)); % ONNX shape is [num_directions, 4*hidden_size, hidden_size], row-major.
            R = iFixWeightsShape(this, R);

            % get biases
            if numel(this.Node.input) >= 4 && ~isempty(this.Node.input{4})
                % If bias name is there, it's an initializer
                B = single(initializerRawData(this.GraphProtoManager, biasName));        % ONNX shape is [num_directions, 8*hidden_size], row-major.
                B = iFixBiasShape(this, B);
            else
                % Bias name is not there. Default to 0
%                 if isBidirectional
%                     B = zeros(8*this.hidden_size,1,'single');
%                 else
                    B = zeros(4*this.hidden_size,1,'single');
%                 end
            end

            % Determine HasStateInputs, and get the H and C initial states:
            HasHInput = false;
            HasCInput = false;
            H = zeros(this.hidden_size,1,'single');            
            if numel(this.Node.input) >= 6 && ~isempty(this.Node.input{6})
                % An initial_h name is there
                hName = this.Node.input{6};
                if isTensorInitializer(this.GraphProtoManager, hName)
                    % initial_h is an initializer. Get it.
                    H = single(initializerRawData(this.GraphProtoManager, hName));    % ONNX shape is [num_directions, batch_size, hidden_size], row-major.
                    [H, stateIssue] = iFixInitialhShape(this, H, "hidden");
                    if isequal(class(stateIssue), 'nnet.internal.cnn.onnx.NodeTranslationWarning')
                        issues = [issues stateIssue];
                    elseif isequal(class(stateIssue), 'nnet.internal.cnn.onnx.NodeTranslationError')
                        issues = [issues stateIssue];
                        Layer = [];
                        return;
                    end
                else
                    % initial_h is an activation tensor. Set its initial
                    % state to zero
                    HasHInput = true;
%                     if isBidirectional
%                         H = zeros(2*this.hidden_size,1,'single');
%                     else
%                         H = zeros(this.hidden_size,1,'single');
%                     end
                end
            end

            C = zeros(this.hidden_size,1,'single');            
            if numel(this.Node.input) >= 7 && ~isempty(this.Node.input{7})
                % An initial_c name is there
                cName = this.Node.input{7};
                if isTensorInitializer(this.GraphProtoManager, cName)
                    % initial_c is an initializer. Get it.
                    C = single(initializerRawData(this.GraphProtoManager, cName));    % ONNX shape is [num_directions, batch_size, hidden_size], row-major.
                    [C, stateIssue] = iFixInitialhShape(this, C, "cell");
                    if isequal(class(stateIssue), 'nnet.internal.cnn.onnx.NodeTranslationWarning')
                        issues = [issues stateIssue];
                    elseif isequal(class(stateIssue), 'nnet.internal.cnn.onnx.NodeTranslationError')
                        issues = [issues stateIssue];
                        Layer = [];
                        return;
                    end
                else
                    % initial_c is an activation tensor. Set its initial
                    % state to zero
                    HasCInput = true;
%                     if isBidirectional
%                         C = zeros(2*this.hidden_size,1,'single');
%                     else
%                         C = zeros(this.hidden_size,1,'single');
%                     end
                end
            end
            if HasHInput && HasCInput
                HasStateInputs = true;
            elseif ~HasHInput && ~HasCInput
                HasStateInputs = false;
            else
                % The node has exactly one of the 2 state inputs
                issues = [issues nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                    message('nnet_cnn_onnx:onnx:BadLSTMInput'))];
                Layer = [];
                return;
            end

            % Determine OutputMode and HasStateOutputs:
            outputUsed = outputUsedInNetwork(this, [1,2,3]); 
            hasOut1 = numel(this.Node.output)>=1 && ~isempty(this.Node.output{1}) && outputUsed(1);
            hasOut2 = numel(this.Node.output)>=2 && ~isempty(this.Node.output{2}) && outputUsed(2);
            hasOut3 = numel(this.Node.output)>=3 && ~isempty(this.Node.output{3}) && outputUsed(3);
            if strcmp(this.TargetNetwork,"dagnetwork")
                % Keep old behavior
                % We can support logical patterns 100, 111, 010, 011.
                if hasOut1 && ~hasOut2 && ~hasOut3
                    OutputMode = 'sequence';
                    HasStateOutputs = false;
                elseif hasOut1 && hasOut2 && hasOut3
                    OutputMode = 'sequence';
                    HasStateOutputs = true;
                elseif ~hasOut1 && hasOut2 && ~hasOut3
                    OutputMode = 'last';
                    HasStateOutputs = false;
                elseif ~hasOut1 && hasOut2 && hasOut3
                    OutputMode = 'last';
                    HasStateOutputs = true;
                else
                    issues = [issues nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                        message('nnet_cnn_onnx:onnx:BadLSTMOutput'))];
                    Layer = [];
                    return;
                end
            else
                % For dlnetwork, we can support all logical patterns except for 000
                if hasOut1
                    OutputMode = 'sequence';
                    HasStateOutputs = hasOut2 || hasOut3;
                elseif ~hasOut1 && (hasOut2 || hasOut3)
                    OutputMode = 'last';
                    HasStateOutputs = hasOut2 || hasOut3;
                else
                    issues = [issues nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                        message('nnet_cnn_onnx:onnx:BadLSTMOutput'))];
                    Layer = [];
                    return;
                end
            end

            % Create DLT layer
            [Layer, constructionIssue] = constructLayer(this, 'lstmLayer', this.LayerName, this.Node, this.hidden_size, 'Name', this.LayerName, ...
                'OutputMode', OutputMode,...
                'HasStateInputs', HasStateInputs,...
                'HasStateOutputs', HasStateOutputs,...
                'StateActivationFunction', hidAct,...
                'GateActivationFunction', iofAct);
            issues = [issues constructionIssue];
            if isempty(Layer)
                return;
            end

            % Set layer weights, biases and state vars
            Layer.InputWeights      = W;
            Layer.RecurrentWeights  = R;
            Layer.Bias              = B;
            if ~HasStateInputs
                Layer.HiddenState       = H;
                Layer.CellState         = C;
            end
        end
    end

    methods(Access=protected)

        function tf = validLabels(this, inputTensorFormats, outputTensorFormats)
            % First input must be 'TBC'
            % 6th and 7th inputs (if given) must be '1BC'
            % First output (if given) must be 'T1BC'
            % Second and third outputs (if given) must be '1BC'
            outputUsed = outputUsedInNetwork(this, [1,2,3]);
            hasIn6 = numel(this.Node.input)>=6 && ~isempty(this.Node.input{6});
            hasIn7 = numel(this.Node.input)>=7 && ~isempty(this.Node.input{7});
            hasOut1 = numel(this.Node.output)>=1 && ~isempty(this.Node.output{1}) && outputUsed(1);
            hasOut2 = numel(this.Node.output)>=2 && ~isempty(this.Node.output{2}) && outputUsed(2);
            hasOut3 = numel(this.Node.output)>=3 && ~isempty(this.Node.output{3}) && outputUsed(3);
            tf = inputTensorFormats(1)=="TBC" && ...
                (~hasIn6 || inputTensorFormats(6)=="1BC") && ...
                (~hasIn7 || inputTensorFormats(7)=="1BC") && ...
                (~hasOut1 || outputTensorFormats(1)=="T1BC") && ...
                (~hasOut2 || outputTensorFormats(2)=="1BC") && ...
                (~hasOut3 || outputTensorFormats(3)=="1BC");
        end

        function issue = iVerifyInitializer(this, key)
            issue = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
            if ~isTensorInitializer(this.GraphProtoManager, key)
                issue =  nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                    message('nnet_cnn_onnx:onnx:LSTMFromOperator'));
            end
        end

        function [H, issue] = iFixInitialhShape(this, H, stateName)
            % ONNX shape is [num_directions, batch_size, hidden_size], row-major.
            % MATLAB order is [hidden_size,1] if forward, [2*hidden_size,1] if bidirectional
            issue = [];
            switch this.direction
                case 'forward'
                    if isequal(numel(H), this.hidden_size)
                        % if batch size is 1
                        H = reshape(H, [this.hidden_size, 1, 1]);                     % Now H is col-major.
                    else
                        % if batch size is not 1
                        if all(H(:) == 0)
                            % if the value are all zeros
                            % reduce the state from [num_directions, batch_size, hidden_size]
                            % to [num_directions, 1, hidden_size]
                            H = reshape(H, this.hidden_size, [], 1);
                            H = H(:,1,1);
                            issue = nnet.internal.cnn.onnx.NodeTranslationWarning(this.Node,...
                                message('nnet_cnn_onnx:onnx:ZeroValueMultiBatchStateLSTM',stateName, ...
                                "["+strjoin(string([this.hidden_size,size(H,2),1]),',')+"]", ...
                                "["+strjoin(string([this.hidden_size,1,1]),',')+"]"));
                        else
                            % throw out error if the values are not all zeros
                            issue = nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                                message('nnet_cnn_onnx:onnx:NoneZeroMultiBatchStateLSTM',stateName));
                        end
                    end
%                 case 'bidirectional'
%                     H = reshape(H, [this.hidden_size, 1, 2]);                     % Now H is col-major.
%                     H = reshape(H, [2*this.hidden_size, 1]);
                otherwise
                    assert(false);
            end
        end

        function B = iFixBiasShape(this, B)
            % ONNX order is Input-Output-Forget-Cell (x2 if bidirectional)
            % MATLAB order is Input-Forget-Cell-Output (x2 if bidirectional)
            iofcRows = reshape(1:4*this.hidden_size, [], 4);                      % Columns of this correspond to ONNX's iofc ordering.
            ifcoRows = iofcRows(:, [1 3 4 2]);                              % Columns of this correspond to DLT's ifco ordering.
            ifcoInd  = ifcoRows(:);                                         % A column vector containing DLT's ifco ordering.
            switch this.direction
                case 'forward'
                    % B is [1, 8*hidden_size], row-major. (Same col-major)
                    B = B(1:4*this.hidden_size) + B(4*this.hidden_size+1:end);	% Combine forward and recurrent biases.
                    B = B(ifcoInd);                               	% B is now in DLT's required ifco order.
                    B = B(:);
%                 case 'bidirectional'
%                     % B is [2, 8*hidden_size], row-major.
%                     B = reshape(B, [8*this.hidden_size, 2]);                      % B is now [8*hiddenSize, 2] col-major (using the "fliplr theorem").
%                     B = B(1:4*this.hidden_size, :) + B(4*this.hidden_size+1:end, :);    % Combine forward and recurrent biases.
%                     B = B(ifcoInd,:);                                       % Each column of B is now in DLT's required ifco order.
%                     B = B(:);                                               % Concatenate the columns.
                otherwise
                    assert(false);
            end
        end

        function [W, issue] = iFixWeightsShape(this, W)
            % ONNX order is Input-Output-Forget-Cell (x2 if bidirectional)
            % MATLAB order is Input-Forget-Cell-Output (x2 if bidirectional)
            iofcRows = reshape(1:4*this.hidden_size, [], 4);                      % Columns of this correspond to ONNX's iofc ordering.
            ifcoRows = iofcRows(:, [1 3 4 2]);                              % Columns of this correspond to DLT's ifco ordering.
            ifcoInd  = ifcoRows(:);                                         % A column vector containing DLT's ifco ordering.
            issue = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
            switch this.direction
                case 'forward'
                    inputSize   = numel(W)/(4*this.hidden_size);                  % W is [1, 4*hidden_size, input_size], row-major.
                    W           = reshape(W, [inputSize, 4*this.hidden_size]);    % W is now [inputSize, 4*hiddenSize] col-major (using the "fliplr theorem").
                    W           = W';                                       % W is now [4*hiddenSize, inputSize].
                    W           = W(ifcoInd, :);                            % Rows of W are now in DLT's required ifco order.
%                 case 'bidirectional'
%                     inputSize   = numel(W)/(8*this.hidden_size);                  % W is [2, 4*hidden_size, input_size], row-major.
%                     W           = reshape(W, [inputSize, 4*this.hidden_size, 2]); % W is now [inputSize, 4*hiddenSize, 2] col-major (using the "fliplr theorem").
%                     W           = reshape(W, [inputSize, 8*this.hidden_size]);    % W is now [inputSize, 8*hiddenSize].
%                     W           = W';                                       % W is now [8*hiddenSize, inputSize].
%                     W           = W([ifcoInd; ifcoInd+4*this.hidden_size], :);    % Rows of W are now in DLT's required ifcoifco order.
                otherwise
                    assert(false);
            end
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

