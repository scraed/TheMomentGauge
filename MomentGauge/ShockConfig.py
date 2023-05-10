import numpy as np
import jax.numpy as jnp
class PhyConstants:
    def __init__(self,Ma = 1.2,velocity_std_ratio=8):

        velocity_std_ratios = np.array( (velocity_std_ratio,)  ).flatten()
        #self.device = torch.device("cpu")
        #self.dtype = torch.float64
        self.pi = np.pi; Pi=self.pi
        
        
        self.Pr = 2/3 # The Prandtl number
        M1=Ma;self.M1 =M1; # The upstream mach number
        l1=1.; self.l1 = l1# The upstream mean free path
        
        Gamma=5./3.;self.Gamma = Gamma # The adiabatic gas exponent
        T1=1.;self.T1 = T1 # The upstream temperature
        v1=np.sqrt(Gamma)*M1;self.v1 = v1 # The upstream velocity
        Rho1=1.;self.Rho1 = Rho1 # The upstream density
        m=1.;self.m = m # The gas molecular mass
        kB=m/T1; self.kB = kB # The Planck Constant, As long as it appears only in terms of kB T /m, it is OK
        self.T2=((2*Gamma*M1**2-(Gamma-1))*((Gamma-1)*M1**2+2))/((Gamma+1)**2*M1**2)*T1 #The downstream temperature
        self.v2=((Gamma-1)*M1**2+2)/((Gamma+1)*M1**2)*v1 #The downstream velocity
        self.Rho2=((Gamma+1)*M1**2)/((Gamma-1)*M1**2+2)*Rho1 #The downstream density
        self.c1=np.sqrt((kB*self.T1)/m) # The upstream speed of sound without Gamma
        self.c2=np.sqrt((kB*self.T2)/m) # The downstream speed of sound without Gamma
        self.Mu=(l1*Rho1*np.sqrt((kB*T1)/m))/np.sqrt(Pi/2) # The upstream viscosity
        self.n1 = Rho1/m
        self.n2 = self.Rho2/m
        self.p1 = T1*self.n1*kB
        self.p2 = self.T2*self.n2*kB
        # Mu = tau n kB T
        self.tau1 = self.Mu/(Rho1*kB*T1/m)  # The relaxiation time in BGK model
        self.A2=1.370347303171576
        
        
        self.Kn = np.sqrt( m/T1/kB )*self.Mu/Rho1
        
        self.estimated_thickness = Shock_Thick_Predict(Ma)
        
        #print(velocity_std_ratios)
        #print(len(velocity_std_ratios))

        if len(velocity_std_ratios) == 1:
            low_x_ratio = velocity_std_ratio
            high_x_ratio = velocity_std_ratio
            high_r_ratio = velocity_std_ratio
        elif len(velocity_std_ratios) == 3:
            low_x_ratio , high_x_ratio , high_r_ratio = velocity_std_ratio
        else:
            assert False, "Could not interprete velocity_std_ratios"

        velocity_lowbound = min( self.v2 - low_x_ratio*(kB*self.T2/m)**0.5, self.v1 - low_x_ratio*(kB*self.T1/m)**0.5 )
        velocity_highbound = max( self.v1 + high_x_ratio*(kB*self.T1/m)**0.5, self.v2 + high_x_ratio*(kB*self.T2/m)**0.5 )
        self.velocity_lowbound = velocity_lowbound
        self.velocity_highbound = velocity_highbound
        
        
        velocity_r_bound = max( high_r_ratio*(kB*self.T1/m)**0.5, high_r_ratio*(kB*self.T2/m)**0.5 )
        self.velocity_r_bound = velocity_r_bound
        #print(velocity_lowbound,velocity_highbound,velocity_r_bound)
"""
def Shock_Thick_Predict(Mtma):
    MSth = 1/(  (((90*Mtma)/(Mtma**2+3))*((Mtma**2-1)**2/(16*Mtma**4-(3+Mtma**2)**2))*((2*np.pi)/15)**0.5)/4 )
    Weakth = 1/( 0.347*(Mtma-1) )
    if Mtma > 2:
        return MSth
    elif Mtma > 1.5:
        return (MSth+Weakth)/2
    else:
        return Weakth  
"""
def Shock_Thick_Predict(Mtma):
    MSth = 1/(  (((90*Mtma)/(Mtma**2+3))*((Mtma**2-1)**2/(16*Mtma**4-(3+Mtma**2)**2))*((2*np.pi)/15)**0.5)/4 )
    Weakth = 1/( 0.347*(Mtma-1) )
    if Mtma > 6.5:
        return 0.9*MSth*(Mtma-2+1)**(0.1)*(Mtma-4+1)**(0.05)*(Mtma-5+1)**(0.05)*(Mtma-6.5+1)**(0.05)
    if Mtma > 5:
        return 0.9*MSth*(Mtma-2+1)**(0.1)*(Mtma-4+1)**(0.05)*(Mtma-5+1)**(0.05)
    if Mtma > 4:
        return 0.9*MSth*(Mtma-2+1)**(0.1)*(Mtma-4+1)**(0.05)
    if Mtma > 2:
        return 0.9*MSth*(Mtma-2+1)**(0.1)
    elif Mtma > 1.1:
        return 0.9*MSth
    else:
        return Weakth  


class Shock_Grid_info:
    def __init__(self, PhyConst, num_cells, domain_ratio = 6):
        x_domain_range = domain_ratio*PhyConst.estimated_thickness
        delta_x = x_domain_range/num_cells; self.delta_x = delta_x
        self.cell_interfaces = jnp.linspace( -x_domain_range/2,x_domain_range/2  ,num_cells+1 )
        self.cell_centers = ( self.cell_interfaces[1:] + self.cell_interfaces[:-1] )/2