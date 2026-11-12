function [Y, HIDDENSTATE, CELLSTATE, numDimsY, numDimsH, numDimsC] = onnxBidirectionalLSTM(X, W, R, B, sequence_lens, initial_h, initial_c, P, stateActivation, gateActivation)

% Copyright 2020-2023 The MathWorks, Inc.

% Prepare arguments for two lstm networks
[H0fwd, H0bwd, C0fwd, C0bwd, WEIGHTSfwd, WEIGHTSbwd, RECURRENTWEIGHTSfwd, RECURRENTWEIGHTSbwd, BIASfwd, BIASbwd]...
    = prepareBidirectionalLSTMArgs(W, R, B, sequence_lens, initial_h, initial_c, P);
% Run the forward lstm
[Yfwd, HIDDENSTATEfwd, CELLSTATEfwd] = lstm(X, H0fwd, C0fwd, WEIGHTSfwd, RECURRENTWEIGHTSfwd, BIASfwd, DataFormat='CBT', StateActivationFunctio=stateActivation, GateActivationFunction=gateActivation);
% Run the backward lstm:
% X has dim ordering CBT. Flip X's time dimension, run a forward lstm on
% it, then flip its output back.
[Ybwd, HIDDENSTATEbwd, CELLSTATEbwd] = lstm(flip(X,3), H0bwd, C0bwd, WEIGHTSbwd, RECURRENTWEIGHTSbwd, BIASbwd, DataFormat='CBT', StateActivationFunction=stateActivation, GateActivationFunction=gateActivation);
Ybwd = flip(Ybwd,3);
% Concatenate the outputs of the two lstms:
% Yfwd has dims CBT. Output Y needs dims CB2T.
[C,B,T] = size(Yfwd, 1:3);
Y = cat(3, reshape(Yfwd,[C B 1 T]), reshape(Ybwd,[C B 1 T]));
% HIDDENSTATEfwd has dims CB. Output HIDDENSTATE needs dims CB2
HIDDENSTATE = cat(3, HIDDENSTATEfwd, HIDDENSTATEbwd);
% CELLSTATEfwd has dims CB. Output CELLSTATE needs dims CB2
CELLSTATE = cat(3, CELLSTATEfwd, CELLSTATEbwd);
% Set output args
Y           = stripdims(Y);
HIDDENSTATE = stripdims(HIDDENSTATE);
CELLSTATE   = stripdims(CELLSTATE);
numDimsY    = 4;   % CB2T
numDimsH    = 3;   % CB2
numDimsC    = 3;   % CB2
end

function [H0fwd, H0bwd, C0fwd, C0bwd, WEIGHTSfwd, WEIGHTSbwd, RECURRENTWEIGHTSfwd, RECURRENTWEIGHTSbwd, BIASfwd, BIASbwd]...
    = prepareBidirectionalLSTMArgs(W, R, B, sequence_lens, initial_h, initial_c, P)
% Prepares arguments for implementing the ONNX biriectional LSTM operator

% All input tensors have reverse-ONNX dimension ordering.
% W has shape [input_size, 4*hidden_size, num_directions].
[input_size, hidden_size4, num_directions] = size(W);
hidden_size = hidden_size4/4;
% initial_h shape is [hidden_size, batch_size, num_directions]
% Each H0 should have shape [hidden_size, batch_size]
if isempty(initial_h)
    H0fwd = dlarray(zeros(hidden_size,1));
    H0bwd = dlarray(zeros(hidden_size,1));
else
    H0fwd = initial_h(:,:,1);
    H0bwd = initial_h(:,:,2);
end
% initial_c shape is [hidden_size, batch_size, num_directions]
% Each C0 should have shape [hidden_size, batch_size]
if isempty(initial_c)
    C0fwd = dlarray(zeros(hidden_size,1));
    C0bwd = dlarray(zeros(hidden_size,1));
else
    C0fwd = initial_c(:,:,1);
    C0bwd = initial_c(:,:,2);
end
% W has shape [input_size, 4*hidden_size, num_directions]. ONNX order of
% hiddens is Input-Output-Forget-Cell. Each matlab WEIGHTS must be a matrix
% of size 4*NumHiddenUnits-by-InputSize. MATLAB order is
% Input-Forget-Cell-Output (ifco).
iofcRows = reshape(1:4*hidden_size, [], 4);                    	% Columns of this correspond to ONNX's iofc ordering.
ifcoRows = iofcRows(:, [1 3 4 2]);                            	% Columns of this correspond to DLT's ifco ordering.
ifcoInd  = ifcoRows(:);                                        	% A column vector containing DLT's ifco ordering.
WEIGHTSfwd  = permute(W(:,:,1), [2 1 3]);                    	% WEIGHTSfwd is [4*hidden_size, inputSize].
WEIGHTSfwd  = WEIGHTSfwd(ifcoInd, :);                         	% Rows of WEIGHTSfwd are now in DLT's required ifco order.
WEIGHTSbwd  = permute(W(:,:,2), [2 1 3]);                    	% WEIGHTSbwd is [4*hidden_size, inputSize].
WEIGHTSbwd  = WEIGHTSbwd(ifcoInd, :);                         	% Rows of WEIGHTSbwd are now in DLT's required ifco order.

% R has shape [hidden_size, 4*hidden_size, num_directions].
% RECURRENTWEIGHTS must be a matrix of size
% 4*NumHiddenUnits-by-NumHiddenUnits. Need to convert iofc to ifco as we
% did for WEIGHTS.
RECURRENTWEIGHTSfwd  = permute(R(:,:,1), [2 1 3]);          	% RECURRENTWEIGHTSfwd is [4*hidden_size, inputSize].
RECURRENTWEIGHTSfwd  = RECURRENTWEIGHTSfwd(ifcoInd, :);       	% Rows of RECURRENTWEIGHTSfwd are now in DLT's required ifco order.
RECURRENTWEIGHTSbwd  = permute(R(:,:,2), [2 1 3]);          	% RECURRENTWEIGHTSbwd is [4*hidden_size, inputSize].
RECURRENTWEIGHTSbwd  = RECURRENTWEIGHTSbwd(ifcoInd, :);       	% Rows of RECURRENTWEIGHTSbwd are now in DLT's required ifco order.

% B has shape [8*hidden_size, num_directions]
% ONNX order is Input-Output-Forget-Cell (x2 if bidirectional)
% MATLAB order is Input-Forget-Cell-Output (x2 if bidirectional)
if isempty(B)
    BIASfwd = dlarray(zeros(4*hidden_size, 1));
    BIASbwd = dlarray(zeros(4*hidden_size, 1));
else
    % B in ONNX is [8*hidden_size, 2] where the first half are the forward
    % biases and the second half are the recurrent biases. In MATLAB these
    % biases are combined.
    BIASfwd = B(1:4*hidden_size, 1) + B(4*hidden_size+1:end, 1);
    BIASfwd = BIASfwd(ifcoInd);                    	% Bfwd is now in DLT's required ifco order.
    BIASbwd = B(1:4*hidden_size, 2) + B(4*hidden_size+1:end, 2);
    BIASbwd = BIASbwd(ifcoInd);                    	% Bbwd is now in DLT's required ifco order.
end
% X shape is [input_size, batch_size, seq_length]. Y will have the same
% dimension ordering as X. That means 'CBT'
% Label the input tensors
H0fwd               = dlarray(stripdims(H0fwd), 'CB');
H0bwd               = dlarray(stripdims(H0bwd), 'CB');
C0fwd               = dlarray(stripdims(C0fwd), 'CB');
C0bwd               = dlarray(stripdims(C0bwd), 'CB');
WEIGHTSfwd          = dlarray(stripdims(WEIGHTSfwd), 'CU');
WEIGHTSbwd          = dlarray(stripdims(WEIGHTSbwd), 'CU');
RECURRENTWEIGHTSfwd = dlarray(stripdims(RECURRENTWEIGHTSfwd), 'CU');
RECURRENTWEIGHTSbwd = dlarray(stripdims(RECURRENTWEIGHTSbwd), 'CU');
BIASfwd             = dlarray(stripdims(BIASfwd(:)), 'C');
BIASbwd             = dlarray(stripdims(BIASbwd(:)), 'C');
end
