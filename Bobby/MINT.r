install.packages('IndepTest')
library(IndepTest)

gen_data <- function(xrange=c(-10,10),num_points=10000,noise_lvl=3,
                    fun_struc="Linear",error_type='Gaussian'){
  x=runif(num_points, xrange[1],xrange[2])
  if (error_type == 'Gaussian'){
    error = rnorm(num_points,0,1)
  } else if (error_type=='Exponential'){
    error= rexp(num_points,rate=1)
  }
  
  if (fun_struc=="Linear"){
    y = x + noise_lvl * error
  } else if (fun_struc == 'Quadratic'){
    y = x^2/2 + noise_lvl *error
  } else if (fun_struc == 'Exponential'){
    y = exp(x) + noise_lvl * error
  } else if (fun_struc == 'Sine'){
    y = sin(x) + noise_lvl *error
  } else if (fun_struc == 'Independent'){
    y = sample(x, size=num_points, replace=FALSE)+noise_lvl*error
  } else if (fun_struc =='SineX'){
    y = sin(x) * x + noise_lvl *error
  } else if (fun_struc == 'Sine/X'){
    y = sin(x)/x  +noise_lvl*error
  }
  
  
  return(cbind(x,y))
}

results <- gen_data(fun_struc='Quadratic',noise_lvl = 10,num_points = 1000)
x <-results[,1]
y <- results[,2]

#0.00990099
quad_p <- MINTav(x,y,K=1:20,B=100)


results <- gen_data(xrange=c(-20,20),fun_struc='SineX',noise_lvl = 5)
x <-results[,1]
y <- results[,2]

sinex_p <- MINTav(x,y,K=1:20,B=100)

results <- gen_data(xrange=c(-5,5),fun_struc='Sine/X',noise_lvl = 0.35)
x <-results[,1]
y <- results[,2]
sine_over_x_p <- MINTav(x,y,K=1:20,B=100)

results <- gen_data(fun_struc='Independent',noise_lvl = 0.35)
x <-results[,1]
y <- results[,2]
indep_p <- MINTav(x,y,K=1:20,B=100)

