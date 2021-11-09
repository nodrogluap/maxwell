# After http://homepages.spa.umn.edu/~willmert/science/ksegments/, updated to Julia 1.1 libraries, data structures and syntax, 
# and writing the data to files for comparison to FLASH implementation.
# Paul Gordon, 2019

# Uncomment the following two lines the first time that you run the program, to ensure you have a plotting back end for Julia
using LinearAlgebra
using DelimitedFiles

(μ,σ) = (4.0,0.5);

x  = 0.0:0.2:10.0

y  = 1 / sqrt(2pi*σ^2) * exp.(-1 .* (x .- μ).^2 / 2σ);

using Random
Random.seed!(1234)	#This is so the data will be the same each time the script is run
Δy = (-1 .+ 2 .* rand(51)) .* 0.2;

data_unscaled = 100 .* (0.2 .+ y .+ Δy);
data = data_unscaled .- (data_unscaled .% 1);
#data = ((data_unscaled .- minimum(data_unscaled) / (maximum(data_unscaled) - minimum(data_unscaled) + 1)));

writedlm("bellman_raw_data_unscaled.txt", data_unscaled, "\n")
writedlm("bellman_raw_data.txt", data, "\n")

function prepare_ksegments(series::Array{Float64,1}, weights::Array{Float64,1})
    N = length(series);

    # Pre-allocate matrices
    wgts = diagm(0 => weights);
    wsum = diagm(0 => weights .* series);
    sqrs = diagm(0 => weights .* series .* series);

    # Also initialize the outputs with sane defaults
    dists = zeros(Float64, N,N);
    means = diagm(0 => series);

    # Fill the upper triangle of dists and means by performing up-right
    # diagonal sweeps through the matrices
    for δ=1:N
        for l=1:(N-δ)
            # l = left boundary, r = right boundary
            r = l + δ;

            # Incrementally update every partial sum
            wgts[l,r] = wgts[l,r-1] + wgts[r,r];
            wsum[l,r] = wsum[l,r-1] + wsum[r,r];
            sqrs[l,r] = sqrs[l,r-1] + sqrs[r,r];

            # Calculate the mean over the range
            means[l,r] = wsum[l,r] / wgts[l,r];
            # Then update the distance calculation. Normally this would have a term
            # of the form
            #   - wsum[l,r].^2 / wgts[l,r]
            # but one of the factors has already been calculated in the mean, so
            # just reuse that.
            dists[l,r] = sqrs[l,r] - means[l,r]*wsum[l,r];        
        end
    end

    return (dists,means)
end

function regress_ksegments(series::Array{Float64,1}, weights::Array{Float64,1}, k::Int)

    # Make sure we have a row vector to work with
    if (length(series) == 1)
        # Only a scalar value
        error("series must have length > 1")
    end

    # Ensure series and weights have the same size
    if (size(series) != size(weights))
        error("series and weights must have the same shape")
    end

    # Make sure the choice of k makes sense
    if (k < 1 || k > length(series))
        error("k must be in the range 1 to length(series)")
    end

    N = length(series);

    # Get pre-computed distances and means for single-segment spans over any
    # arbitrary subsequence series(i:j). The costs for these subsequences will
    # be used *many* times over, so a huge computational factor is saved by
    # just storing these ahead of time.
    (one_seg_dist,one_seg_mean) = prepare_ksegments(series, weights);

    # Keep a matrix of the total segmentation costs for any p-segmentation of
    # a subsequence series[1:n] where 1<=p<=k and 1<=n<=N. The extra column at
    # the beginning is an effective zero-th row which allows us to index to
    # the case that a (k-1)-segmentation is actually disfavored to the 
    # whole-segment average.
    k_seg_dist = zeros(Float64, k, N+1);
    # Also store a pointer structure which will allow reconstruction of the
    # regression which matches. (Without this information, we'd only have the
    # cost of the regression.)
    k_seg_path = zeros(Int, k, N);

    # Initialize the case k=1 directly from the pre-computed distances
    k_seg_dist[1,2:end] = one_seg_dist[1,:];

    # Any path with only a single segment has a right (non-inclusive) boundary
    # at the zeroth element.
    for i=1:N
        k_seg_path[1,i] = 0;
    end
    # Then for p segments through p elements, the right boundary for the (p-1)
    # case must obviously be (p-1).
    for i in 1:k
        k_seg_path[i,i] = k - 1;
    end

    # Now go through all remaining subcases 1 < p <= k
    for p=2:k
        # Update the substructure as successively longer subsequences are
        # considered.
        for n=p:N
            # Enumerate the choices and pick the best one. Encodes the recursion
            # for even the case where j=1 by adding an extra boundary column on the
            # left side of k_seg_dist. The j-1 indexing is then correct without
            # subtracting by one since the real values need a plus one correction.
            choices = Array{Float64}(undef, n);
            for i=1:n
                choices[i] = k_seg_dist[p-1, i] + one_seg_dist[i, n];
            end

            (bestval,bestidx) = findmin(choices);

            # Store the sub-problem solution. For the path, store where the (p-1)
            # case's right boundary is located.
            k_seg_path[p,n] = bestidx - 1;
            # Then remember to offset the distance information due to the boundary
            # (ghost) cells in the first column.
            k_seg_dist[p,n+1] = bestval;
        end
    end

    # Eventual complete regression
    reg = zeros(Float64, size(series));

    # Now use the solution information to reconstruct the optimal regression.
    # Fill in each segment reg(i:j) in pieces, starting from the end where the
    # solution is known.
    rhs = length(reg);
    for p=k:-1:1
		println(rhs);
        # Get the corresponding previous boundary
        lhs = k_seg_path[p,rhs];

        # The pair (lhs,rhs] is now a half-open interval, so set it appropriately
        for i=lhs+1:rhs
            reg[i] = one_seg_mean[lhs+1,rhs];
        end

        # Update the right edge pointer
        rhs = lhs;
    end
	println(rhs);

    return reg
end

# Even weighting for all data points in the series
wght = ones(size(data));

# Run the regression
regression = regress_ksegments(data, wght, 6);

# Write the regression breakpoints to a file
writedlm("bellman_segmentated_data.txt", unique(regression), "\n")

# Write the raw regression points to a file
writedlm("bellman_segmentated_raw_data.txt", regression, "\n")

