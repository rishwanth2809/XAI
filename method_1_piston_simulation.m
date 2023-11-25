clc;
clear all;
close all;

%% Problem definition

n = 1000;

% Generate data using LHS
M = 30 + (60-30) * lhsdesign(n, 1);
k = 1000 + (5000-1000) * lhsdesign(n, 1);
S = 0.005 + (0.020-0.005) * lhsdesign(n, 1);
Po = 90000 + (110000-90000) * lhsdesign(n, 1);
Vo = 0.002 + (0.010-0.002) * lhsdesign(n, 1);
To = 340 + (360-340) * lhsdesign(n, 1);
Ta = 290 + (296-290) * lhsdesign(n, 1);

% Calculate A and V
A = Po .* S + 19.62 .* M - (k .* Vo ./ S);
V = (S ./ (2.*k)) .* (sqrt(A.^2 + 4.*k.*(Po.*Vo./To).*Ta) - A);

% Calculate the piston simulation output
C = 2 * pi * sqrt(M ./ (k + S.^2 .* (Po.*Vo./To) .* (Ta./V.^2)));

% Concatenate into a data matrix
data = [M k S Po Vo To Ta C];


%% Dataset
comp_names = {'M', 'k', 'S', 'Po', 'Vo', 'To', 'Ta', 'C'};
sData = som_data_struct(data,'comp_names',comp_names); 
sData = som_normalize(sData,'range');

% Initializing SOM Map Codebook Vectors (Linear Initialization)
[sMap]= modifiedsom_lininit1(sData,'lattice','hexa','msize',[20,20]);

% Training SOM
[sMap,sTrain] = modifiedsom_batchtrain(sMap,sData,'sample_order','ordered','trainlen',500,...
     'radius_ini', 1.0, 'radius_fin',0.9);

% Denormalizing the data
sMap=som_denormalize(sMap,sData);   
sData=som_denormalize(sData,'remove');

% Visualization of SOM results (U Matrix and Component Planes)
figure(1) 
som_show(sMap, 'umat','all','comp', 'all');

% iSOM Grid in function space (using just the first three components for visualization)
figure(2)
som_grid(sMap,'coord',sMap.codebook(:,[1 2 8]),'label',sMap.labels,'labelcolor','c','labelsize',10, 'marker','o','MarkerColor','k',...
    'MarkerSize',7,'linecolor', 'k');
hold on, scatter3(data(:,1),data(:,2),data(:,8),20,'ro','filled');
xlabel('M')
ylabel('k')
zlabel('C')


%% Normalize Data
norm_data = som_normalize(sMap, 'range');

%% Extract codebook vectors
v1 = norm_data.codebook(:,1);  % M
v2 = norm_data.codebook(:,2);  % k
v3 = norm_data.codebook(:,3);  % S
v4 = norm_data.codebook(:,4);  % Po
v5 = norm_data.codebook(:,5);  % Vo
v6 = norm_data.codebook(:,6);  % Ta
v7 = norm_data.codebook(:,7);  % To
vC = norm_data.codebook(:,8);  % y (output)

%% Compute slopes for each variable
M_slope = calculate_slope(v1);
k_slope = calculate_slope(v2);
S_slope = calculate_slope(v3);
Po_slope = calculate_slope(v4);
Vo_slope = calculate_slope(v5);
Ta_slope = calculate_slope(v6);
To_slope = calculate_slope(v7);

%% Project gradients of vC onto the slopes of each feature
[M_pos, M_neg] = projected_slope(vC, M_slope);
[k_pos, k_neg] = projected_slope(vC, k_slope);
[S_pos, S_neg] = projected_slope(vC, S_slope);
[Po_pos, Po_neg] = projected_slope(vC, Po_slope);
[Vo_pos, Vo_neg] = projected_slope(vC, Vo_slope);
[Ta_pos, Ta_neg] = projected_slope(vC, Ta_slope);
[To_pos, To_neg] = projected_slope(vC, To_slope);

%% Aggregate all results
results.positive_correls = [M_pos, k_pos, S_pos, Po_pos, Vo_pos, Ta_pos, To_pos];
results.negative_correls = [M_neg, k_neg, S_neg, Po_neg, Vo_neg, Ta_neg, To_neg];

%% Plotting
features = {'M', 'k', 'S', 'Po', 'Vo', 'Ta', 'To'};
figure;

subplot(2, 1, 1);
hold on;
barh(1:7, -results.negative_correls, 'r');
barh(1:7, results.positive_correls, 'g');
yticks(1:7); yticklabels(features);
grid on; hold off;

subplot(2, 1, 2);
barh(1:7, abs(results.positive_correls) + abs(results.negative_correls), 'b');
yticks(1:7); yticklabels(features);
grid on;


%%




function slope = calculate_slope(codebook_vec)
    data_ = reshape(codebook_vec, [20, 20])
    data_ = flipud(data_);
    [dZdx, dZdy] = gradient(data_);
    avg_dZdx = mean(dZdx(2:end-1, 2:end-1), 'all');
    avg_dZdy = mean(dZdy(2:end-1, 2:end-1), 'all');
    slope = [avg_dZdx, avg_dZdy];
end



function [positive_correl, negative_correl] = projected_slope(codebook_vec, input_slope)

    % Reshape and flip the codebook vector
    data_ = reshape(codebook_vec, [20, 20]);
    data_ = flipud(data_);
    
    % Calculate gradient
    [dZdx, dZdy] = gradient(data_);

    % Normalize the gradients
    grad_magnitudes = sqrt(dZdx.^2 + dZdy.^2);
    dZdx = dZdx ./ grad_magnitudes;
    dZdy = dZdy ./ grad_magnitudes;

    disp([dZdx, dZdy])

    % Normalize the input slope to get a unit vector
    magnitude = norm(input_slope); 
    unit_slope = input_slope / magnitude;
    
    % Initialize variables to hold the sum of vectors
    pos_sum = [0, 0];
    neg_sum = [0, 0];
    
    % Project dZdx and dZdy onto the input slope
    for i = 2:19
        for j = 2:19
            gradient_vec = [dZdx(i, j), dZdy(i, j)]; % make it a row vector
            dot_product = dot(gradient_vec, unit_slope);
            
            % Project gradient vector onto the unit slope
            projected_vec = (dot_product) * unit_slope;
            
            if dot_product > 0
                pos_sum = pos_sum + projected_vec;
            else
                neg_sum = neg_sum + projected_vec;
            end
        end
        
    end
    
    % Compute the magnitudes for vectors in each direction
    positive_correl = norm(pos_sum)/324;
    negative_correl = norm(neg_sum)/324;
    
end
