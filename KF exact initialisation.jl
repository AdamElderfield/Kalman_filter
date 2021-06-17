using Base: thread_notifiers
# Kalman filter exact initialisation 

using DataFrames
using Optim 
using LinearAlgebra


import XLSX



# ---------------------------------------------------------
# Function 1 - initialisation
# Function 1 - Kalman filter recursions
# Function 2 - Log Likelihood estimation
# Function 3 - Kalman smoother recursions
# ---------------------------------------------------------
# ALT-J then ALT-O to resart REPL

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Mod
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  

mutable struct dlm  
    FF
    H
    Q
    R
    A
    k
    n
    cur_ξ_hat
    cur_Σ_hat
end


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Initialisation
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Function to initialse filter
#   1) Diffuse
#   2) Exact diffuse


# Sets up dlm, given system matrices
 # ξ_t = FFξ_(t-1) + Qw_t
 #  y_t = A'x_t + H'ξ_t + Rv_t 

 # TODO: Expand to inlcude x and y?
function dlm(FF,H, Q, R, A)
    k = size(H, 2)   # Number of y variables (columns of H as it is transposed)
    n = size(H, 1)   # Number of states (row of H as it is transposed
    ξ = n == 1 ? zero(eltype(FF)) : zeros(n)                    # IF n == 1 then 0, else zeros[n,1]
    Σ = n == 1 ? zero(eltype(FF)) : zeros(n, n)                 # IF n == 1 then 0, else zeros[n,n]   
    return dlm(FF, H, Q, R, A, k, n, ξ, Σ)
end


# User defined initialisation
function initialise!(m::dlm, ξ, Σ)
    m.cur_ξ_hat = ξ
    m.cur_Σ_hat = Σ
    Nothing
end

# Exact limit 

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Kalman filter
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ξ_t = Fξ_(t-1) + w_t
# y_t = A'x_t + H'ξ_t + v_t
# Assumes states have been initialised

function KalmanFilter(m::dlm,y::AbstractArray, x = "NA")
    #m =mod1
    FF, A, H, Q, R = m.FF, m.A, m.H, m.Q, m.R
      
    k, n, = m.k, m.n
    
    # *********************************
    # Initialisation of matrices
    # *********************************

    # Number of time periods
    T = size(y,1)

    # Filtered mean and variance 
    ξ_f = zeros(T,n)
    Σ_f = zeros(n,n,T)
    
    # Predicted mean and variance 
    ξ_p = zeros(T,n)
    Σ_p = zeros(n,n,T)

    if x == "NA"
        x = zeros(T,1)
    end    
    
    for t= 1:T
       
        # ******************************
        # Prediction step 
        # ******************************
        #t = 1
        #--------------------------------
        # Conditional initialses the filter with priors if t = 1
        #--------------------------------
        if t ==1

            ξ_p[t,:] = FF[n,n]*m.cur_ξ_hat 
        
            Σ_p[:,:,t] = FF[n,n]*m.cur_Σ_hat*transpose(FF[n,n]) .+ Q[n,n]

        else    

            ξ_p[t,:] = FF[:,:]*ξ_f[t-1,:]     
            
            Σ_p[:,:,t] = FF[:,:]*Σ_f[:,:,t-1]*transpose(FF[:,:]) .+ Q[:,:]

        end

        # Ensure Y vector is y[vars,time]
        if m.k > 1
            reshape(y[t], m.k, 1)
        end

        # Matrix size error here!
        # Prediction error
        Hp = transpose(H)
        Ap = transpose(A)

        prediction_error = (y[t].-Ap[:,:]*x[t].-Hp[:,:]*ξ_p[t,:])

        # Prediction variance
        HΣHR = transpose(H[:,:])*Σ_p[:,:,t]*H[:,:] .+ R[:,:]

        # ******************************
        # Filtered step 
        # ******************************
        
        # Kalman Gain
        Gain = (Σ_p[:,:,t]*H[:,:])/(HΣHR[:,:])

        # Filtered mean
        ξ_f[t,:]  = ξ_p[t,:] .+ Gain[:,:]*prediction_error[:,:]

        # Filtered variance
        Σ_f[:,:,t] = Σ_p[:,:,t] .- Gain[:,:]*H[:,:]*Σ_p[:,:,t]

    end    
  
    return ξ_p,Σ_p,ξ_f,Σ_f

end


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Kalman smoother
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Tests
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# Nile river data for testing
dat =  DataFrame(XLSX.readtable("C:/Users/aelde/OneDrive/Documents/GitHub/Tutorials/State space models/NileDat.xlsx","Sheet1")...)


# Sets up dlm, given system matrices
 # ξ_t = FFξ_(t-1) + Qw_t
 #  y_t = A'x_t + H'ξ_t + Rv_t 

 A = [0.0]
 H = [1.0; 0.0]
 FF = [1.0 1.0;0.0  1.0]
 Q = [1469.1 0.0; 0.0   100.0]
 R = [15099.0]

mod1= dlm(FF,H,Q,R,A)
initialise!(mod1,[0.0 ;0.0],[100000 0; 0 1000])
a =KalmanFilter(mod1,dat[:,1])
a[3]
plot([a[3],y])