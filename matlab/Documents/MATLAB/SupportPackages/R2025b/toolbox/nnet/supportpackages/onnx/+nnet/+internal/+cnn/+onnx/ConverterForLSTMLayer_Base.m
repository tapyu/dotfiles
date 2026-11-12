classdef ConverterForLSTMLayer_Base < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a lstmLayer into ONNX

    % Supported IO as of 23a:
    % CBT(sequence) --> CBT
    % CBT(last) --> CB
    % CB(sequence) --> CB
    % CB(last) --> CB
    % Hidden and Cell inputs and outputs, when present, are always CB.

    % Copyright 2018-2022 The MathWorks, Inc.

    methods
        function this = ConverterForLSTMLayer_Base(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end

        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] = toOnnx(...
                this, nodeProto, TensorNameMap, TensorLayoutMap, passedInputWeights, passedRecurrentWeights)  
            % Note that inputWeights are passed as function arguments in
            % this class. The derived classes ConverterForLSTMLayer and
            % ConverterForLSTMProjectedLayer pass them.
            import nnet.internal.cnn.onnx.*

            existingNodeNames       = {nodeProto.name};            
            [lstmNodeName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            lstmNodeName                = makeUniqueName(existingNodeNames, lstmNodeName);
            NNTLayer                = this.NNTLayer;
            hasSequenceOutput       = isequal(NNTLayer.OutputMode, 'sequence');

            inputTensorNames = mapTensorNames(this, this.InputLayerNames, TensorNameMap);

            inputTensorLayout   = TensorLayoutMap(inputTensorNames{1});

            % Unsqueeze if inputTensorLayout is 'CB'
            if inputTensorLayout == "nc"
                unsqueezeAxis                 = 0;
                unsqueezeName                 = [inputTensorNames{1} '_Unsqueeze'];
                unsqueezeName                 = makeUniqueName(existingNodeNames, unsqueezeName);
                unsqueezeInput                = {inputTensorNames{1}};
                unsqueezeOutput               = {unsqueezeName};
                [unsqueezeNode, unsqueezeInit]  = createNodeProto(this, 'Unsqueeze', unsqueezeName, unsqueezeInput, unsqueezeOutput, unsqueezeAxis);
                lstmInputTensorName           = unsqueezeName;
                hasSequenceOutput             = false;
            elseif inputTensorLayout == "snc"
                lstmInputTensorName = inputTensorNames{1};
                unsqueezeNode = [];
                unsqueezeInit = [];
            end

            %Set outputTensorLayout
            if hasSequenceOutput
                lstmOutputTensorLayout  = 'snc';
            else
                lstmOutputTensorLayout  = 'nc';
            end

            % Make the LSTM inputs
            lstmInput               = {...
                lstmInputTensorName,... % X
                [lstmNodeName '_W'],...
                [lstmNodeName '_R'],...
                [lstmNodeName '_B'],...
                '',...                      % sequence_Lens
                [lstmNodeName '_initial_h'],...
                [lstmNodeName '_initial_c']
                };
            if NNTLayer.HasStateInputs
                warning(message('nnet_cnn_onnx:onnx:UnsupportedRNNStateInputs', this.NNTLayer.Name))
                lstmInput(6:7) = inputTensorNames(2:3);
            end

            % Create nodes to transform the state inputs, if present.
            [StateInputPreprocessingNodes, StateInputPreprocessingInits, lstmInput] = createStateInputPreprocessingNodes(...
                this, NNTLayer, lstmNodeName, lstmInput);

            lstmInput = mapTensorNames(this, lstmInput(:)', TensorNameMap);

            % Make the output name appear as either the sequence output or
            % the 'last' output as appropriate. Include state outputs if
            % necessary.
            if NNTLayer.HasStateOutputs
                lstmOutputTensorName    = [lstmNodeName '/out'];
                hiddenOutputTensorName  = [lstmNodeName '/hidden'];
                cellOutputTensorName    = [lstmNodeName '/cell'];
                if hasSequenceOutput
                    lstmOutput      = {lstmOutputTensorName hiddenOutputTensorName cellOutputTensorName};
                else
                    lstmOutput      = {'' hiddenOutputTensorName cellOutputTensorName};
                end
            elseif hasSequenceOutput
                lstmOutputTensorName    = lstmNodeName;
                lstmOutput         = {lstmOutputTensorName};          % Output is Y=onnxName
            else
                lstmOutputTensorName    = lstmNodeName;
                lstmOutput         = {'', lstmOutputTensorName};      % Output is Y='', Y_h=onnxName
            end

            % Set attributes
            iofAct	= iTranslateActivationFunction(NNTLayer.GateActivationFunction);
            cellAct	= iTranslateActivationFunction(NNTLayer.StateActivationFunction);
            hidAct	= cellAct;

            % Make the lstm node
            [lstmNode, lstmInits] = createNodeProto(this, 'LSTMForward', lstmNodeName, lstmInput, lstmOutput, ...
                NNTLayer.InputSize, hasSequenceOutput, ...
                iofAct, cellAct, hidAct, passedInputWeights, passedRecurrentWeights, NNTLayer.Bias, ...
                NNTLayer.NumHiddenUnits);

            % Follow LSTM with Squeeze operators to reshape the outputs
            % Squeeze main output (Y or Y_h):
            if hasSequenceOutput
                squeezeAxis = 1;
            else
                squeezeAxis = 0;
            end
            squeezeName                 = [lstmNodeName '_Squeeze'];
            squeezeName                 = makeUniqueName([existingNodeNames, {lstmNodeName}], squeezeName);
            squeezeInput                = {lstmOutputTensorName};
            squeezeOutput               = {squeezeName};
            [squeezeNode, squeezeInit]  = createNodeProto(this, 'Squeeze', squeezeName, squeezeInput, squeezeOutput, squeezeAxis);
            lstmOutputTensorName        = squeezeName;

            % Maybe Squeeze state outputs from 1NC to NC
            squeezeCellNode = [];
            squeezeCellInit = [];
            squeezeHiddenNode = [];
            squeezeHiddenInit = [];
            if NNTLayer.HasStateOutputs
                % Squeeze the cell output
                squeezeCellName         = [lstmNodeName '_SqueezeCell'];
                squeezeCellName         = makeUniqueName([existingNodeNames, {lstmNodeName}], squeezeCellName);
                squeezeCellInput        = {cellOutputTensorName};
                squeezeCellOutput       = {squeezeCellName};
                [squeezeCellNode, squeezeCellInit]  = createNodeProto(this, 'Squeeze', squeezeCellName, squeezeCellInput, squeezeCellOutput, 0);
                cellOutputTensorName    = squeezeCellName;
                % Squeeze the Y_h output if it isn't the main output:
                if hasSequenceOutput
                    squeezeHiddenName         = [lstmNodeName '_SqueezeHidden'];
                    squeezeHiddenName         = makeUniqueName([existingNodeNames, {lstmNodeName}], squeezeHiddenName);
                    squeezeHiddenInput        = {hiddenOutputTensorName};
                    squeezeHiddenOutput       = {squeezeHiddenName};
                    [squeezeHiddenNode, squeezeHiddenInit]  = createNodeProto(this, 'Squeeze', squeezeHiddenName, squeezeHiddenInput, squeezeHiddenOutput, 0);
                    hiddenOutputTensorName    = squeezeHiddenName;
                end
            end

            % Collect the appropriate set of nodes and initializers
            nodeProto = [nodeProto unsqueezeNode StateInputPreprocessingNodes lstmNode squeezeNode squeezeCellNode squeezeHiddenNode];
            parameterInitializers = [unsqueezeInit StateInputPreprocessingInits lstmInits squeezeInit squeezeCellInit squeezeHiddenInit];

            networkInputs           = [];
            networkOutputs          = [];

            % Determine layer output port names. These are the names used
            % by the MATLAB representation of the layer
            onnxNamePort = this.NNTLayer.Name;
            lstmOutputPort = this.NNTLayer.Name;
            if NNTLayer.HasStateOutputs
                lstmOutputPort = [this.NNTLayer.Name '/out'];
                hiddenOutputPort = [this.NNTLayer.Name '/hidden'];
                cellOutputPort = [this.NNTLayer.Name '/cell'];
            end

            % Update maps
            TensorNameMap(onnxNamePort) = lstmOutputTensorName;
            TensorNameMap(lstmOutputPort) = lstmOutputTensorName;
            TensorLayoutMap(lstmOutputTensorName) = lstmOutputTensorLayout;
            if NNTLayer.HasStateOutputs
                TensorNameMap(hiddenOutputPort) = hiddenOutputTensorName;
                TensorLayoutMap(hiddenOutputTensorName) = 'nc';
                TensorNameMap(cellOutputPort) = cellOutputTensorName;
                TensorLayoutMap(cellOutputTensorName) = 'nc';
                if ~hasSequenceOutput
                    TensorNameMap(onnxNamePort) = hiddenOutputTensorName;
                    TensorNameMap(lstmOutputPort) = hiddenOutputTensorName;
                    TensorLayoutMap(lstmOutputTensorName) = 'nc';
                end
            end

            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end

        function [nodes, inits, lstmInputNames] = createStateInputPreprocessingNodes(this, NNTLayer, lstmNodeName, lstmInputNames)
            % Create nodes to transform the state inputs prior to LSTM
            % node. Update lstmInputNames with the names output by these
            % nodes.
            import nnet.internal.cnn.onnx.*
            if NNTLayer.HasStateInputs
                % There are state inputs. Insert unsqueeze nodes.

                % Unsqueeze initial_h from BC to 1BC
                hUnsqNodeName           = [lstmNodeName '_Unsqueeze_Hidden_Input'];
                [hUnsqNode, hUnsqInit]  = createNodeProto(this, 'Unsqueeze', hUnsqNodeName, lstmInputNames(6), {hUnsqNodeName}, 0);

                % Unsqueeze initial_c from BC to 1BC
                cUnsqNodeName           = [lstmNodeName '_Unsqueeze_Cell_Input'];
                [cUnsqNode, cUnsqInit]  = createNodeProto(this, 'Unsqueeze', cUnsqNodeName, lstmInputNames(7), {cUnsqNodeName}, 0);

                nodes               = [hUnsqNode, cUnsqNode];
                inits               = [hUnsqInit, cUnsqInit];
                lstmInputNames{6}   = hUnsqNodeName;
                lstmInputNames{7}   = cUnsqNodeName;
            else
                % There are no state inputs. Insert nodes that gather the
                % batchSize from X and expand the state initializers to
                % their required shape using Tile. We want to create 1BH in
                % onnx, given Hx1 matlab data and the ONNX X input of shape
                % TBC. The following ONNX code has been carefully chosen so
                % it works for all opsets>=6:
                %
                % S = Shape(X)					                            S is rank 1, int64  = [T B C].
                % B = Gather(S, init([1]), axis=0)			                Make indices [1] (rank 1) to get rank 1 output.
                % repeats = Concat(init([1]), B, init([1]), axis=0).	    Build the vector [1,B,1]
                % newH = Tile(H, repeats)                                   Turn shape [1,1,H] into [1,B,H]
                % newC = Tile(C, repeats)                                   Turn shape [1,1,H] into [1,B,H]

                % Shape(X) node:
                sNodeName = [lstmNodeName '_Shape1'];
                [sNode, ~]  = createNodeProto(this, 'Shape', sNodeName, lstmInputNames(1), {sNodeName});

                % Gather node:
                gNodeName = [lstmNodeName '_Gather1'];
                gIndicesName = [lstmNodeName '_Gather1Indices'];
                gInit = nnet.internal.cnn.onnx.makeTensorProtoOfType(gIndicesName, [1], 1, TensorProto_DataType.INT64);
                [gNode, ~]  = createNodeProto(this, 'Gather', gNodeName, {sNodeName, gIndicesName}, {gNodeName}, 0);

                % Concat node:
                cNodeName = [lstmNodeName '_Concat1'];
                cConst1Name = [lstmNodeName '_ConcatConst1'];
                cConst1Init = nnet.internal.cnn.onnx.makeTensorProtoOfType(cConst1Name, [1], 1, TensorProto_DataType.INT64);
                [cNode, ~]  = createNodeProto(this, 'Concat', cNodeName, {cConst1Name, gNodeName, cConst1Name}, {cNodeName}, 0);

                % Tile(H) node:
                tNodeName1 = [lstmNodeName '_TileH'];
                [tNode1, ~]  = createNodeProto(this, 'Tile', tNodeName1, {lstmInputNames{6}, cNodeName}, {tNodeName1});

                % Tile(C) node:
                tNodeName2 = [lstmNodeName '_TileC'];
                [tNode2, ~]  = createNodeProto(this, 'Tile', tNodeName2, {lstmInputNames{7}, cNodeName}, {tNodeName2});

                % Create initializers for the raw initial_h and initial_c
                % input tensors, of shape [1,1,nH].
                HInit       = makeTensorProtoOfType(lstmInputNames{6}, [1,1,NNTLayer.NumHiddenUnits], NNTLayer.HiddenState, TensorProto_DataType.FLOAT);
                CInit       = makeTensorProtoOfType(lstmInputNames{7}, [1,1,NNTLayer.NumHiddenUnits], NNTLayer.CellState, TensorProto_DataType.FLOAT);

                % Return the nodes and inits and update the LSTM input names
                nodes               = [sNode, gNode, cNode, tNode1, tNode2];
                inits               = [gInit, cConst1Init, HInit, CInit];
                lstmInputNames{6}   = tNodeName1;
                lstmInputNames{7}   = tNodeName2;
            end
        end
    end
end

function s = iTranslateActivationFunction(s)
switch s
    case 'sigmoid'
        s = 'Sigmoid';
    case 'softsign'
        s = 'Softsign';
    case 'tanh'
        s = 'Tanh';
    case 'relu'
        s = 'Relu';
    case 'hard-sigmoid'
        s = 'HardSigmoid';
    otherwise
        error(message('nnet_cnn_onnx:onnx:InvalidActivationFunction', s, layerName));
end
end
