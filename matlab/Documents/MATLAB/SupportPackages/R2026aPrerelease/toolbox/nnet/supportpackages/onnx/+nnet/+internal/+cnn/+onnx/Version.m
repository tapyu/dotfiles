classdef Version < int32
    % enum Version {
    %   _START_VERSION = 0;
    %   IR_VERSION_2017_10_10 = 0x0000000000000001;
    %   IR_VERSION_2017_10_30 = 0x0000000000000002;
    %   IR_VERSION_2017_11_3 = 0x0000000000000003;
    %   IR_VERSION_2019_1_22 = 0x0000000000000004;
    %   IR_VERSION_2019_3_18 = 0x0000000000000005;
    %   IR_VERSION_2019_9_19 = 0x0000000000000006;
    %   IR_VERSION_2020_5_8 = 0x0000000000000007;
    %   IR_VERSION_2021_7_30 = 0x0000000000000008;
    %   IR_VERSION = 0x0000000000000009;
    % }
    
    %   Copyright 2019-2024 The MathWorks, Inc.

    enumeration
        START_VERSION           (0)
        IR_VERSION_2017_10_10   (1)
        IR_VERSION_2017_10_30   (2)
        IR_VERSION_2017_11_3    (3)
        IR_VERSION_2019_1_22    (4)
        IR_VERSION_2019_3_18    (5)
        IR_VERSION_2019_9_19    (6)
        IR_VERSION_2020_5_8     (7)
        IR_VERSION_2021_7_30    (8)
        IR_VERSION              (9)
    end
end