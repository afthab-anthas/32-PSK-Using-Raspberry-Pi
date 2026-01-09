
% MATLAB 32-PSK DSP FINAL CODE DEC 1
% all comments are written by our group members for easy understanding

clc; clear; close all;
port = 5174;
server = tcpserver("0.0.0.0", port);
disp("Server ready");

Tb = 1e-6;        % symbol duration T which is 1/10^-6 (1 microsecond) it means that the symbol will be drawn in a wave with 1ms gap
nb = 100;         % samples per symbol (to calculate each symbol, you take 100 samples from the continous signal)
Fc = 10e6;        % carrier frequency is set to 10mhz (the AC current wave that is used to transport data)
h  = 0.7;         % channel attenuation or strength fading of the signal to simulate real world channel
noise_var = 1;    % AWGN variance which determines the strength of the noise added to our ideal signal
dt   = Tb/nb;	  % sampling interval which tells us the time interval between taking two samples (duration / no of samples)
t_sym = 0:dt:(Tb-dt);  % a structure to store time axis (starts from 0 - sampling interval - final symbol) (satrt - step - end)   
phi1 = cos(2*pi*Fc*t_sym); % basis function 1 which will carry our symbol - cosine wave with 10mhz frequency and the time axis details (symbol data)
phi2 = sin(2*pi*Fc*t_sym); % same basis function 2, but here we use sine wave

%% the whole point of N = 2 (2 basis functions) is because our constellation data is 2D and the signals only carry 1D data. Hence we have cosine wave to carry x value and sine wave to carry y value
%% x,y - real,imaginary - inphase,quadrature
%% we use cosine and sine waves beacuse they both are orthogonal signals (perpendicular to eachother). If you multiply the recieved signal by sine or cosine, the opposite wave dissapears leaving only the respective multipled wave.


while true
    
    while server.NumBytesAvailable == 0 % constantly checking for data
        pause(0.05);
    end
    rawData = read(server, server.NumBytesAvailable, "string"); % stores the received data into raw variable
    
    
    % tryin to catch json in an if else type structure which says try (code) and if it fails store the err and run end (code)
    try
        jsonData = jsondecode(rawData); % converts raw (string) data into matlab structured array 
        bits = jsonData.flat_bits; % the json data sent from rpi is structured which has the bits inside flat-bits ad hence we are accessign that data
    catch err
        disp("JSON decode failed."); continue;
    end
    bits = bits(:).'; % structuring the data into a matrix with the rows filled with the bits received
    fprintf("%d bits received\n", length(bits));
    
    % DSP STRUCTURE 
    
    symbols_nat = vector_encode_32psk(bits); % data used to store list of decimal numbers (converted from bits of 5)s
    if isempty(symbols_nat)
        disp("Not enough bits."); continue;
    end
    
    symbols_gray = gray_map(symbols_nat); % mapping the symbols to gray coded symbols
    CONST = generate_constellation_32psk(); % generates ideal constellation coordinates
    X_tx = CONST(symbols_gray + 1, :).'; % structures the data into a matrix where the coordinates from the const var are selected by searching for the symbols + 1 (to adjust index = 0)
    [s_t, t_total] = modulate_signal(X_tx, phi1, phi2, Tb); % s_t is the physical signal whose inputs are the coordinates, cosine func, sine func, symbol duration (speed)
    y_t = awgn_channel_time(s_t, h, noise_var); % adds attenuation and AWGN to our ideal s_t signal
    R_rx = demodulate_matched(y_t, phi1, phi2, t_sym, Tb); % R_rx hold the demodulated signal using matched filter %
    detected_gray = ml_detect_vectors(R_rx, CONST); % performs maximum likelihood detection using euclidean distance %
    detected_nat = inv_gray_map(detected_gray); % reverses gray coding to convert the symbols into its natural order
    detected_bits = symbols_to_bits_32psk(detected_nat); % converts the symbols back to a bitstream
    [BER, SER] = compute_errors_32psk(symbols_nat, detected_nat, bits, detected_bits); 
    bit_err = sum(bits(1:length(detected_bits)) ~= detected_bits); % counts the bit error keeping the detected bit length in mind and checks if the first element is not equl to (~=) the detected bis

    fprintf("SER: %.4f | BER: %.4f | Bit Errors: %d\n", SER, BER, bit_err);

    %% ============================================================
    % GENERATE PLOTS & CONVERT TO BASE64
    % ============================================================
    fprintf("Generating and encoding plots...\n");
    

    % We generate 4 separate plots, save them to memory, and encode to Base64
    img_time = fig_to_base64(@() plot_time_domain_32psk(bits, detected_bits, s_t, y_t, Tb, nb));
    img_ideal = fig_to_base64(@() plot_const_ideal(CONST));
    img_tx    = fig_to_base64(@() plot_tx_syms(X_tx));
    img_rx    = fig_to_base64(@() plot_rx_syms(R_rx));
    img_final = fig_to_base64(@() plot_rx_const(CONST, R_rx, X_tx));

    %% ============================================================
    % SEND JSON RESPONSE
    % ============================================================
    response.status = "success";
    response.SER = SER;
    response.BER = BER;
    response.total_bits = length(bits);
    response.bit_errors = bit_err;
    
    response.plot_ideal = img_ideal;
    response.plot_tx    = img_tx;
    response.plot_rx    = img_rx;
    response.plot_final = img_final;
    response.plot_time = img_time;

    jsonResponse = jsonencode(response);
    write(server, jsonResponse, "string");
    fprintf("Results sent back to Pi.\n----------------------------------------------------\n");
end

%% ============================================================
%                   HELPER FUNCTIONS
% ============================================================

% --- 1. BASE64 PLOT CONVERTER ---
function b64 = fig_to_base64(plotFunc)
    f = figure('Visible','off', 'Color','w', 'Position', [0 0 500 400]);
    plotFunc(); % Run the specific plotting logic
    
    % Save to temp file
    tempName = [tempname '.png'];
    exportgraphics(f, tempName, 'Resolution', 100); 
    
    % Read bytes and encode
    fid = fopen(tempName, 'rb');
    raw = fread(fid, inf, '*uint8');
    fclose(fid);
    b64 = matlab.net.base64encode(raw);
    
    % Cleanup
    delete(tempName);
    close(f);
end

% --- 2. PLOTTING LOGIC (Separated for JSON packaging) ---
function plot_const_ideal(CONST)
    blue = [0 102 255]/255;
    scatter(CONST(:,1), CONST(:,2), 90, blue, 'filled');
    title("1. Ideal Constellation"); xlabel('I'); ylabel('Q');
    grid on; axis equal; xlim([-1.4 1.4]); ylim([-1.4 1.4]);
end

function plot_tx_syms(X_tx)
    green = [0 204 102]/255;
    scatter(X_tx(1,:), X_tx(2,:), 70, green, 'filled');
    title("2. Transmitted Symbols"); xlabel('I'); ylabel('Q');
    grid on; axis equal; xlim([-1.4 1.4]); ylim([-1.4 1.4]);
end

function plot_rx_syms(R_rx)
    red = [255 51 51]/255;
    scatter(R_rx(1,:), R_rx(2,:), 70, red, 'filled');
    title("3. Rx Symbols (Noisy)"); xlabel('I'); ylabel('Q');
    grid on; axis equal; xlim([-1.4 1.4]); ylim([-1.4 1.4]);
end

function plot_rx_const(CONST, R_rx, X_tx)
    % Define Colors
    black = [0 0 0];          % Ideal Reference
    pink  = [1 0.2 0.6];      % Received (Rx)
    green = [0 0.7 0.3];      % Transmitted (Tx)
    gray  = [0.6 0.6 0.6];    % Error Vector
    cyan  = [0.6 0.8 1];      % Boundaries

    hold on;

    % 1. Draw Boundaries (Capture handle of the first one for legend)
    M = 32; radius = 1.6;
    % Plot the first one and save handle "h1"
    h1 = plot([0 radius], [0 0], '--', 'Color', cyan, 'LineWidth', 0.5); 
    % Plot the rest
    for k = 0:M-1
        theta = (2*pi*k)/M + (pi/M); 
        plot([0 radius*cos(theta)], [0 radius*sin(theta)], '--', 'Color', cyan, 'LineWidth', 0.5);
    end

    % 2. Draw Error Vectors (Capture handle of the first one "h2")
    num_syms = size(X_tx, 2);
    if num_syms > 0
        h2 = plot([X_tx(1,1) R_rx(1,1)], [X_tx(2,1) R_rx(2,1)], '-', 'Color', gray);
        for k = 2:num_syms
            plot([X_tx(1,k) R_rx(1,k)], [X_tx(2,k) R_rx(2,k)], '-', 'Color', gray);
        end
    else
        h2 = plot(0,0,'-','Color',gray); % Dummy if no symbols
    end

    % 3. Plot Points (Capture handles h3, h4, h5)
    h3 = scatter(CONST(:,1), CONST(:,2), 100, black, 'filled'); 
    h4 = scatter(X_tx(1,:), X_tx(2,:), 60, green, 's', 'filled'); 
    h5 = scatter(R_rx(1,:), R_rx(2,:), 40, pink, 'filled'); 

    title("4. Decision Boundaries & Error Vectors"); 
    xlabel('I'); ylabel('Q');
    grid on; axis equal; 
    xlim([-1.6 1.6]); ylim([-1.6 1.6]);
    
    % Force legend to use specific handles and stay INSIDE to maximize size
    legend([h1, h2, h3, h4, h5], ...
           {'Boundary', 'Noise Vector', 'Ideal Ref', 'Tx Sent', 'Rx Recv'}, ...
           'Location', 'best', 'FontSize', 8); 
    hold off;
end

function plot_time_domain_32psk(bits, detected_bits, s_t, y_t, Tb, nb)

    % In 32-PSK, 5 bits fit into 1 Symbol Duration (Tb)
    % nb = 100 samples per symbol. Therefore, samples per bit = 100 / 5 = 20.
    samples_per_bit = nb / 5; 
    
    % 2. Create Square Wave for Input Bits (Tx) 
    L = length(bits);
    digit_tx = [];
    for n = 1:L
        if bits(n) == 1
            sig = ones(1, samples_per_bit);
        else
            sig = zeros(1, samples_per_bit);
        end
        digit_tx = [digit_tx sig];
    end
    
    % 3. Create Square Wave for Output Bits 
    L_rx = length(detected_bits);
    digit_rx = [];
    for n = 1:L_rx
        if detected_bits(n) == 1
            sig = ones(1, samples_per_bit);
        else
            sig = zeros(1, samples_per_bit);
        end
        digit_rx = [digit_rx sig];
    end

    % 4. Create Time Vectors
    % Time axis for signals (already exists as length of s_t)
    t_signal = linspace(0, (L/5)*Tb, length(s_t));
    % Time axis for bits (must match the length of the square wave)
    t_bits_tx = linspace(0, (L/5)*Tb, length(digit_tx));
    t_bits_rx = linspace(0, (L_rx/5)*Tb, length(digit_rx));

    % 5. Plotting (4-Subplot Time Domain Structure)
    subplot(4,1,1);
    plot(t_bits_tx, digit_tx, 'LineWidth', 1.5);
    axis([0 (L/5)*Tb -0.2 1.2]); % Set Y-axis to clearly show 0 and 1
    title('1. Digital Input Signal at Tx'); xlabel('Time(s)'); ylabel('Volts');
    grid on;

    subplot(4,1,2);
    plot(t_signal, s_t);
    xlim([0 (L/5)*Tb]);
    title('2. Modulated/Transmitted Signal'); xlabel('Time(s)'); ylabel('Volts');
    grid on;

    subplot(4,1,3);
    plot(t_signal, y_t);
    xlim([0 (L/5)*Tb]);
    title('3. Received Signal (Noisy)'); xlabel('Time(s)'); ylabel('Volts');
    grid on;

    subplot(4,1,4);
    plot(t_bits_rx, digit_rx, 'LineWidth', 1.5);
    axis([0 (L_rx/5)*Tb -0.2 1.2]);
    title('4. Digital Output Signal at Rx'); xlabel('Time(s)'); ylabel('Volts');
    grid on;
end

%% --- 3. DSP LOGIC FUNCTIONS ---
function symbols = vector_encode_32psk(bits)
    bits = bits(:).'; N = length(bits); numSym = floor(N / 5); % here the numsym is telling us how many symbols can we have (floor is round off function)
    if numSym == 0, symbols = []; return; end
    used = bits(1:numSym*5); M = reshape(used, 5, numSym);
    weights = [16; 8; 4; 2; 1]; symbols = weights.' * double(M);
end
function g = gray_map(s), g = bitxor(s, floor(s/2)); end
function b = inv_gray_map(g)
    b = zeros(size(g));
    for k = 1:numel(g), val=g(k); res=val; 
        while val>0, val=bitshift(val,-1); res=bitxor(res,val); end; b(k)=res; 
    end
end
function C = generate_constellation_32psk(), M=32; k=0:M-1; th=2*pi*k/M; C=[cos(th)', sin(th)']; end
function [s_t, t] = modulate_signal(X, p1, p2, Tb)
    nb=length(p1); N=size(X,2); s_t=zeros(1,N*nb);
    for k=1:N, s_t((k-1)*nb+1:k*nb) = X(1,k)*p1 + X(2,k)*p2; end
    t=0:Tb/nb:(N*Tb-Tb/nb);
end
function y = awgn_channel_time(s,h,v), y=h*s + sqrt(v)*randn(size(s)); end
function R = demodulate_matched(y, p1, p2, t, Tb)
    nb=length(p1); num=floor(length(y)/nb); R=zeros(2,num);
    for k=1:num, seg=y((k-1)*nb+1:k*nb); 
        R(:,k)=[(2/Tb)*trapz(t,seg.*p1); (2/Tb)*trapz(t,seg.*p2)]; end
end
function d = ml_detect_vectors(R, C)
    [~,N]=size(R); d=zeros(1,N); 
    for k=1:N, [~,idx]=min(sum((C - R(:,k).').^2, 2)); d(k)=idx-1; end
end
function bits = symbols_to_bits_32psk(s)
    bits=zeros(1,length(s)*5); idx=1;
    for x=s, bits(idx:idx+4)=bitget(x,5:-1:1); idx=idx+5; end
end
function [B,S] = compute_errors_32psk(txS, rxS, txB, rxB)
    S=sum(txS~=rxS)/length(txS); L=min(length(txB),length(rxB)); B=sum(txB(1:L)~=rxB(1:L))/L;
end

