#include <algorithm>

/* The idea here is to have anchor points in the middle of the warp path that must be passed through, which
   lets us reduce the cost matrix size a lot since we can compute a separate, smaller cost matrix for each of the 
   between-anchor sections and then stitch the warp paths together (hence stitch_dtw function below). If you had a streaming DTW implementation to find the
   location of the query in a large subject based on collinearity (within a band), you'd already know the anchor points,
   i.e. the collinear blocks. This saves us effort compared to sparseDTW or other global warp path efficiency methods. 
   In a way you could consider this a "seed-and-extend" approach to DTW. */

#include <algorithm>

void traceback(float *matrix, int nx, int ny, int xoffset, long yoffset, std::vector< std::pair<int,long> > &path){
    int i = nx-1;
    int j = ny-1;
    path.push_back(std::make_pair(xoffset+i,yoffset+j)); // end anchor
    while ((i > 0) && (j > 0)){
        // 3 conditions below are equivalent to min((matrix[i-1, j-1], matrix[i, j-1], matrix[i-1, j]))
        if (matrix[i-1+(j-1)*nx] <= matrix[i+(j-1)*nx] && matrix[i-1+(j-1)*nx] <= matrix[(i-1)+j*nx]){ // diagonal move is preferred (i.e. White-Neely step scoring)
            --i;
            --j;
        }
        else if (matrix[i+(j-1)*nx] <= matrix[i-1+(j-1)*nx] && matrix[i+(j-1)*nx] <= matrix[(i-1)+j*nx])
            --j;
        else
            --i;
        path.push_back(std::make_pair(i+xoffset,j+yoffset));
        //std::cout << matrix[i+j*nx] << "/";
    }
    while(i > 0){
        path.push_back(std::make_pair((--i)+xoffset,yoffset));
    }
    while(j > 0){
        path.push_back(std::make_pair(xoffset,(--j)+yoffset));
    }
    // flip the path back to ascending order
    std::reverse(path.begin(), path.end());
    //std::cout << std::endl;
}

double euclidean_dtw(float *x, int nx, float *y, int ny, int xoffset, long yoffset, std::vector< std::pair<int,long> > &path){
    float max = std::numeric_limits<float>::max();
    
    float *accumulated_cost_matrix = (float *) malloc(sizeof(float)*nx*ny);
    if(accumulated_cost_matrix == 0){
      std::cerr << "Could not allocate cost matrix for DTW of size (" << nx << "," << ny << "), aborting" << std::endl;
      exit(1);
    }
    for(int i = 1; i < nx; ++i){
      accumulated_cost_matrix[i] = max;
    }
    for(int i = 1; i < ny; ++i){
      accumulated_cost_matrix[i*nx] = max;
    }
    accumulated_cost_matrix[0] = abs(x[0]-y[0]);
    for(int i = 1; i < nx; ++i){
        for(int j = 1; j < ny; ++j){
            // 3 conditions below are equivalent to min((matrix[i-1, j-1], matrix[i, j-1], matrix[i-1, j]))
            if(accumulated_cost_matrix[i-1+(j-1)*nx] <= accumulated_cost_matrix[i+(j-1)*nx] && accumulated_cost_matrix[i-1+(j-1)*nx] <= accumulated_cost_matrix[i-1+j*nx]){
                accumulated_cost_matrix[i+j*nx] = abs(x[i]-y[j]) + accumulated_cost_matrix[i-1+(j-1)*nx];
            }
            else if(accumulated_cost_matrix[i+(j-1)*nx] <= accumulated_cost_matrix[i-1+(j-1)*nx] && accumulated_cost_matrix[i+(j-1)*nx] <= accumulated_cost_matrix[i-1+j*nx]){
                accumulated_cost_matrix[i+j*nx] = abs(x[i]-y[j]) + accumulated_cost_matrix[i+(j-1)*nx];
            }
            else{
                accumulated_cost_matrix[i+j*nx] = abs(x[i]-y[j]) + accumulated_cost_matrix[i-1+j*nx];
            }
        }
    }
    traceback(accumulated_cost_matrix, nx, ny, xoffset, yoffset, path);
    float cost = accumulated_cost_matrix[nx*ny-1];
    free(accumulated_cost_matrix);
    return cost;
}

/* returned path vector is pairs <query_position,subject_position> */
double stitch_dtw(std::vector< std::pair<int,long> > input_warp_anchors, float *query, float *subject, std::vector< std::pair<int,long> > &output_warp_path){
   //TODO: align head and tail outside the anchors?
   double total_cost = 0;
   for(int i = 0; i < input_warp_anchors.size()-1; ++i){
       std::vector< std::pair<int,long> > local_path;
       int size_vector = (input_warp_anchors[i+1].first-input_warp_anchors[i].first)*1.5;
       // std::cerr << "size of vector in stitch is: " << size_vector << std::endl;
       if(size_vector < 0){
         std::cerr << "Error, blocks are not in order. Revieved size less than zero. Aborting." << std::endl;
         exit(1);
       }
       local_path.reserve(size_vector);
       //std::cout << "Stitching query (" << input_warp_anchors[i].first << "," << input_warp_anchors[i+1].first << ") with subject (" << input_warp_anchors[i].second << "," << input_warp_anchors[i+1].second-1 << "), cost ";
       double cost = euclidean_dtw(&query[input_warp_anchors[i].first], input_warp_anchors[i+1].first-input_warp_anchors[i].first, 
                     &subject[input_warp_anchors[i].second], input_warp_anchors[i+1].second-input_warp_anchors[i].second, 
                     input_warp_anchors[i].first, input_warp_anchors[i].second, local_path);
       //std::cout << cost << std::endl;
       total_cost += cost;
       output_warp_path.insert(output_warp_path.end(), local_path.begin(), local_path.end());
   } 
   return total_cost;
}
