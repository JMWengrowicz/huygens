import numpy as np
from scipy import signal as sig
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib import colors
import distributions as dist
from high_math import *
from copy import deepcopy
import warnings
from scipy.signal import convolve2d as conv2





# import sys
# sys.path.append(r'C:\Users\yonatanwe\PycharmProjects\beta_version_modules')

class FieldArray:

    def __init__(self, array=np.array([0]), lam=1e-9, pix=13.5e-6, center=(0, 0), label=''):
        self.field = array
        self.Nx = np.size(array, 1)
        self.Ny = np.size(array, 0)
        self.lam = lam  # wavelength
        self.pix = pix  # pixel size. Currently must be the same size for x and y (only square pixels and not rectangles).
        self.center = center  # measured in [m]. This is the coordinates of the center of the matrix.
        self.edge = [(float(center[0] - pix * self.Nx / 2), float(center[0] + pix * self.Nx / 2)), (float(center[1] - pix * self.Ny / 2), float(center[1] + pix * self.Ny / 2))]
        if label is None:
            label = ''
        self.label = label
        return

    def __add__(self, other):
        out = deepcopy(self)
        out.field += other.field
        if self.Nx!=other.Nx or self.Ny!=other.Ny or self.lam!=other.lam or self.pix!=other.pix:
            warnings.warn('WARNING: Field arrays with different pix or lam has been added')

        return out

    def __sub__(self, other):
        out = deepcopy(self)
        out.field -= other.field
        if self.Nx!=other.Nx or self.Ny!=other.Ny or self.lam!=other.lam or self.pix!=other.pix:
            warnings.warn('WARNING: Field arrays with different pix or lam has been substructed')

        return out

    def __mul__(self, other):
        out = deepcopy(self)
        out.field *= other.field
        if self.Nx!=other.Nx or self.Ny!=other.Ny or self.lam!=other.lam or self.pix!=other.pix:
            warnings.warn('WARNING: Field arrays with different pix or lam has been multiplied')

        return out


    def huygens(self, L, scale=1, shift=(0, 0), newN=(0, 0)):

        # The scale here is for the pixel size of the output matrix relative to the input matrix.
        # newN is the size of the output matrix
        # shift is the coordinate shift of the center of the output matrix relative to the input center
        # L is propagation length

        if newN[0] == 0:
            Nx = self.Nx
        else:
            Nx = newN[0]
        if newN[1] == 0:
            Ny = self.Ny
        else:
            Ny = newN[1]

        k = 2*np.pi/self.lam
        # edge = pix*self.N/2
        tx = segment_around(self.center[0], self.pix, self.Nx)
        ty = segment_around(self.center[1], self.pix, self.Ny)
        # t = np.linspace(-self.edge, self.edge, self.N)
        Tx = segment_around(self.center[0] + shift[0], self.pix * scale, Nx)
        Ty = segment_around(self.center[1] + shift[1], self.pix * scale, Ny)
            # = np.linspace(-self.edge, self.edge, newN) * scale
        # x0, x = np.meshgrid(t, t * scale)
        # y0, y = np.meshgrid(t, t * scale)
        x0, x = np.meshgrid(tx, Tx)
        y0, y = np.meshgrid(ty, Ty)
        temp_x = self.pix * np.exp(-1j * k * ((x - x0) ** 2) / (2 * L))
        temp_y = self.pix * np.exp(-1j * k * ((y - y0) ** 2) / (2 * L))
        temp_x = np.transpose(temp_x)

        out = deepcopy(self)
        out.field = 1j * np.exp(-1j * k * L) * temp_y @ self.field @ temp_x / (L * self.lam)
        out.Nx = np.size(out.field, 1)
        out.Ny = np.size(out.field, 0)
        # out.pix = self.pix * scale
        out.pix = self.pix * scale  # * self.N/newN
        out.center = (self.center[0] + shift[0], self.center[1] + shift[1])
        out.edge = [(float(out.center[0] - out.pix * out.Nx / 2), float(out.center[0] + out.pix * out.Nx / 2)), (float(out.center[1] - out.pix * out.Ny / 2), float(out.center[1] + out.pix * out.Ny / 2))]

        return out

    def get_energy(self):
        return np.sum(np.abs(self.field)**2)

    def get_max(self):
        running_idx = np.argmax(np.abs(self.field))
        # print(np.unravel_index(np.abs(self.field).argmax(), self.field.shape))
        x = running_idx%self.Nx
        y = int(np.floor(running_idx/self.Nx))
        val = self.field[y, x]
        # print(np.abs(self.field[y,x]))
        # print(x)
        # print(y)
        return val, y, x

    def show(self, n=None, label=None):

        fig = plt.figure(n)

        img = abs(self.field)**2
        # plt.imshow(img, extent=[-self.edge, self.edge, -self.edge, self.edge])
        # edge_x = float(self.edge)

        plt.imshow(img, extent=[self.edge[0][0], self.edge[0][1], self.edge[1][1], self.edge[1][0]], aspect='auto')
        plt.ticklabel_format(style='sci', scilimits=(0, 0))
        plt.colorbar()
        if label is None:
            label = self.label
        label += ''
        plt.title(label)
        # plt.xlim((-self.edge,self.edge))
        # plt.ylim((-self.edge, self.edge))
        # plt.show(block=True)
        # plt.savefig(str(n))
        return

    def show_log(self, n=None, label=None):

        fig = plt.figure(n)

        # img = np.log10(abs(self.field)**2)
        img = abs(self.field)**2
        # plt.imshow(img, extent=[-self.edge, self.edge, -self.edge, self.edge])
        # edge = float(self.edge)
        plt.imshow(img, extent=[self.edge[0][0], self.edge[0][1], self.edge[1][1], self.edge[1][0]], aspect='auto', norm=colors.LogNorm())
        plt.ticklabel_format(style='sci', scilimits=(0,0))
        plt.colorbar()
        if label is None:
            label = self.label
        label += ' log'
        plt.title(label)
        # plt.xlim((-self.edge,self.edge))
        # plt.ylim((-self.edge, self.edge))
        # plt.show(block=True)
        # plt.savefig(str(n))
        return

    def show_phase(self, n=None, label=None, only_phase=False):

        fig = plt.figure(n)

        img_hue = np.angle(self.field)
        img_sat = np.ones(self.field.shape)
        if only_phase:
            img_val = np.ones(self.field.shape)
        else:
            img_val = np.abs(self.field)

        img_hue += np.pi
        img_hue /= 2 * np.pi
        # img_val -= np.min(img_val)
        img_val /= np.max(img_val)

        img = np.dstack((img_hue, img_sat, img_val))
        # plt.imshow(hsv_to_rgb(img), extent=[-self.edge, self.edge, -self.edge, self.edge], cmap='hsv')
        # edge = float(self.edge)
        if only_phase:
            plt.imshow(np.angle(self.field), extent=[self.edge[0][0], self.edge[0][1], self.edge[1][1], self.edge[1][0]],
                       cmap='hsv', aspect='auto')
        else:
            plt.imshow(hsv_to_rgb(img), extent=[self.edge[0][0], self.edge[0][1], self.edge[1][1], self.edge[1][0]],
                       cmap='hsv', aspect='auto')
        plt.ticklabel_format(style='sci', scilimits=(0,0))
        # plt.imshow(hsv_to_rgb(img), cmap='hsv')
        plt.colorbar()
        plt.clim(-np.pi, np.pi)

        if label is None:
            label = self.label
        label += ' phase'
        plt.title(label)
        # plt.xlim((-self.edge,self.edge))
        # plt.ylim((-self.edge, self.edge))
        # plt.show(block=True)
        # plt.savefig(str(n))
        return

    def show_cs(self, n=None, axis=0, r=0.5, label=None):

        fig = plt.figure(n)

        if label is None:
            label = self.label
        label += ' cross-section'


        if axis == 0:
            N = self.Ny
            N_orth = self.Nx
        elif axis == 1:
            N = self.Nx
            N_orth = self.Ny

        rp = round(r * (N_orth-1))
        cs = abs(self.field[rp, :]) ** 2

        plt.plot(np.linspace(self.edge[1-axis][0], self.edge[1-axis][1], N), cs, label=label)
        plt.ticklabel_format(style='sci', scilimits=(0, 0))
        # plt.show(block=True)
        # plt.savefig(str(n))
        return

    def show_cs_logy(self, n=None, axis=0, r=0.5, norm=True, label=None):

        fig = plt.figure(n)

        if label is None:
            label = self.label
        label += ' cross-section'

        if axis == 0:
            N = self.Ny
            N_orth = self.Nx
        elif axis == 1:
            N = self.Nx
            N_orth = self.Ny

        rp = round(r * (N_orth - 1))
        cs = abs(self.field[rp, :]) ** 2

        if norm:
            cs /= max(cs)

        plt.semilogy(np.linspace(self.edge[1-axis][0], self.edge[1-axis][1], N), cs, label=label)
        plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='x')
        # plt.show(block=True)
        # plt.savefig(str(n))
        return

    def show_ang_mean(self, n=None, norm=True, label=None):

        fig = plt.figure(n)

        if label is None:
            label = self.label
        label += ' mean'

        out, PSF, STD = angular_integration(np.abs(self.field)**2)
        if not norm:
            PSF = out[:, 3]

        plt.plot(self.pix*out[:, 0], PSF, label=label)
        plt.ticklabel_format(style='sci', scilimits=(0, 0))
        # plt.show(block=True)
        # plt.savefig(str(n))
        return

    def show_ang_mean_logy(self, n=None, norm=True, label=None):

        fig = plt.figure(n)

        if label is None:
            label = self.label
        label += ' mean'

        out, PSF, STD = angular_integration(np.abs(self.field)**2)
        if not norm:
            PSF = out[:, 3]

        plt.semilogy(self.pix*out[:, 0], PSF, label=label)
        plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='x')
        plt.grid(True, which='major', color='dimgray', ls='-')
        plt.grid(True, which='minor', color='darkgray', ls='--')
        # plt.show(block=True)
        # plt.savefig(str(n))
        return

    def show_ang_int(self, n=None, norm=True, label=None):

        fig = plt.figure(n)

        if label is None:
            label = self.label
        label += ' integral'

        out, PSF, STD = angular_integration(np.abs(self.field)**2)

        if norm:
            out[:, 1] = out[:, 1]/max(out[:, 1])

        plt.plot(self.pix*out[:, 0], out[:, 1], label=label)
        plt.ticklabel_format(style='sci', scilimits=(0, 0))
        # plt.show(block=True)
        # plt.savefig(str(n))
        return

    def show_ang_int_logy(self, n=None, norm=True, label=None):

        fig = plt.figure(n)

        if label is None:
            label = self.label
        label += ' integral'

        out, PSF, STD = angular_integration(np.abs(self.field)**2)
        if norm:
            out[:, 1] = out[:, 1] / max(out[:, 1])

        plt.semilogy(self.pix * out[:, 0], out[:, 1], label=label)
        plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='x')
        plt.grid(True, which='major', color='dimgray', ls='-')
        plt.grid(True, which='minor', color='darkgray', ls='--')
        # plt.show(block=True)
        # plt.savefig(str(n))
        return

    def add_mask(self, mask):
        out = deepcopy(self)
        out.field *= mask
        return out

    def pass_lens(self, focal_length=2e-2):
        # t = np.linspace(-self.edge, self.edge, self.N)
        tx = segment_around(self.center[0], self.pix, self.Nx)
        ty = segment_around(self.center[1], self.pix, self.Ny)
        x, y = np.meshgrid(tx, ty)
        mask = np.exp(1j * 2 * np.pi * (x ** 2 + y ** 2) / (2 * self.lam * focal_length))
        out = deepcopy(self)
        out.field *= mask
        return out


def sphere_wave(z0=-1, pix=13.5e-6, N=2000, x0=0, y0=0, lam=1e-9, amplitude=1, label=None):

    k = 2*np.pi/lam
    t = np.linspace(-N*pix/2, N*pix/2, N)
    x, y = np.meshgrid(t, t)
    phase = ((x - x0) ** 2 + (y - y0) ** 2) * k / (2 * z0)
    f_out = np.exp(1j * phase)*amplitude*pix/(2*np.sqrt(z0 ** 2 + (x - x0) ** 2 + (y - y0) ** 2)*np.sqrt(np.pi))

    out = FieldArray(array=f_out, lam=lam, pix=pix, label=label)
    return out


def gaussian_beam(x_width=5e-3, y_width=5e-3, center=(0, 0), pix=8e-6, N=(1000, 1000), lam=1.03e-6, label=None):
    x_grid, y_grid = pix * np.mgrid[-0.5*N[1]:0.5*N[1], -0.5*N[0]:0.5*N[0]]
    exponent = -((x_grid - center[1]) ** 2 / (2 * (x_width/2) ** 2) + (y_grid - center[0]) ** 2 / (2 * (y_width/2) ** 2))
    power = np.exp(exponent)
    field = np.sqrt(power) * np.exp(1j*0)

    out = FieldArray(array=field, lam=lam, pix=pix, label=label)

    return out


def lens(N=1000, radius=5e-2, focal_length=2e-2, lam=600e-9):
    t = np.linspace(-radius, radius, N)
    x, y = np.meshgrid(t, t)
    mask = np.exp(1j * 2 * np.pi * (x ** 2 + y ** 2) / (2 * lam * focal_length))
    return mask


def pinhole(N=2000, r=1, T=0):
    k = int(np.ceil(2000/N))

    x, y = np.meshgrid(np.linspace(-1, 1, k*N), np.linspace(-1, 1, k*N))
    A0 = x**2+y**2 <= r**2
    A = np.array(A0, dtype="complex_")
    # B = np.reshape(A, k, N, k, N)
    B = A.reshape(k, N, k, N, order='F')
    # B = A.reshape(k, N, k, N, order='F').copy()
    C = np.sum(np.sum(B, 0), 1)/(k**2)
    mask = C*(1-T)+T
    # mask = np.reshape(C, [N, N])
    return mask


def nap(n=1000, D=40e-6, d=100e-9, N=2000, golden=False, apod=False, T=0, r=1, theta0=0):
    edge = D/r/2
    gr = np.linspace(-edge, edge, N)
    pix = gr[2]-gr[1]
    Nsub = int(np.ceil(d/pix))
    mask = np.zeros([N+2*Nsub, N+2*Nsub], dtype="complex_")
    subhole = pinhole(N=Nsub, r=d / pix / Nsub)
    R_amp = D / 2 / edge * (N / 2)
    dist_func = dist.uniform
    if apod:
        R_amp *= 1.08
        dist_func = dist.hamming

    if golden:
        golden_ratio = (1 + np.sqrt(5)) / 2
        R = np.linspace(1 / n, 1, n)
        R = dist_func(R)

        R *= R_amp
        theta = np.array(range(n)) * 2 * np.pi * golden_ratio + theta0
        x11 = np.round(R * np.sin(theta))
        y11 = np.round(R * np.cos(theta))

        xind = x11+N/2+Nsub
        yind = y11+N/2+Nsub
        xind = xind.astype('int32')
        yind = yind.astype('int32')
        mask[xind, yind] = 1

    else:
        factor_collisions = 4
        R = np.random.uniform(0, 1, n*factor_collisions)
        R = dist_func(R)
        if n == 1:
            R = np.zeros(factor_collisions)
        R *= R_amp
        theta = np.random.uniform(0, 2 * np.pi, n*factor_collisions)
        x11, y11 = pol2cart(R, theta)
        # x = np.round(((x1 / edge) + 1) * (N - 1) / 2 + Nsub)
        # y = np.round(((y1 / edge) + 1) * (N - 1) / 2 + Nsub)
        x = x11 + N / 2 + Nsub
        y = y11 + N / 2 + Nsub
        x = x.astype('int32')
        y = y.astype('int32')
        number_holes = 0
        i = -1
        double_subhole = pinhole(N=2 * Nsub + 1, r=1)

        while (number_holes < n) and (i < factor_collisions * n):

            i = i + 1
            ROI = mask[int(x[i]) - Nsub:int(x[i]) + Nsub+1, int(y[i]) - Nsub: int(y[i]) + Nsub+1]
            if np.sum(ROI * double_subhole) == 0:
                mask[x[i], y[i]] += 1
                number_holes = number_holes + 1
            elif np.sum(ROI * double_subhole) < 0:
                print('you are idiot')

        print('Number of relocations: ' + str(i - n + 1))
        if i >= factor_collisions * n:
            print('WARNING: number of subholes is ' + str(number_holes) + ' instead of ' + str(n))

    mask = sig.convolve2d(mask[Nsub:-Nsub, Nsub:-Nsub], subhole, mode='same')
    mask = mask * (1 - T) + T

    return mask


def knife_edge(N=2000, r=0.5):
    # This func creates NxN matrix, where its left 'half' is '1', and right half is '0'
    # In: N - Matrix size, r=location of knife edge (if 0.5=> in the middle, if 0 all is 1, if 1 all is 0
    # example: from matplotlib import pyplot as plt
    #          from optics import knife_edge
    #          plt.imshow(knife_edge(100,0.2))
    #          plt.colorbar()
    #          plt.imshow(np.transpose(1-knife_edge(100,0.2)))  # To receive knife_edge in Y axis, '1' up and '0' down
    maskX, maskY = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    mask = np.array((maskX >= r), dtype="complex_")
    if r == 1: mask *= 0  # For r==1 we would like all mask to be ==0
    return mask



def tg(Nx=2000, Ny=2000, N=None, period=100e-9, aper=100e-9, structure='full', ygap=15e-9, hight=10e-6, width=10e-6, spitz_pos=0):
    # This function has not yet finished
    if N is not None:
        Nx = N
        Ny = N

    spacex = 2 * period
    spacey = 2 * (aper + ygap)
    pix_size = max(hight/Ny, width/Nx)
    # H = pix_size * (Ny-1)
    # W = pix_size * (Nx-1)

    # xax = np.arange(-W / 2, W / 2, pix_size)
    # yax = np.arange(-H / 2, H / 2, pix_size)
    # xax = np.linspace(-W / 2, W / 2, Nx)
    # yax = np.linspace(-H / 2, H / 2, Ny)
    xax = segment_around(0, pix_size, Nx)
    yax = segment_around(0, pix_size, Ny)
    X, Y = np.meshgrid(xax, yax)

    if structure == 'full':
        density_factor = 1
        # spitz_pos = .5

    elif structure == 'checkers':
        density_factor = 2
        # spitz_pos = 0

    else:
        return 'unknown structure'

    fixed_aper = spitz_pos*1/2*np.arcsin(spitz_pos)+np.cos(1/2*np.arcsin(spitz_pos))**2
    # A = round(aper / pix_size)
    A = round(fixed_aper * aper / pix_size)
    lam = round(period / pix_size)

    xline = ((np.diff((X-min(xax)) % spacex, 1, 1) < 0) + density_factor * (np.diff(np.mod((X-min(xax))+spacex/2, spacex), 1, 1) < 0))
    yline = ((np.diff((Y-min(yax)) % spacey, 1, 0) < 0) + density_factor * (np.diff(np.mod((Y-min(yax))+spacey/2, spacey), 1, 0) < 0))

    xline = np.column_stack((xline, np.zeros(Ny))) * (np.abs(X) <= (width - spacex + pix_size) / 2)
    yline = np.vstack((yline, np.zeros(Nx))) * (np.abs(Y) <= (hight - spacey + pix_size) / 2)

    pos_mat = (xline == yline) * (xline != 0)

    xax2 = np.linspace(-np.pi, np.pi, lam)/2
    yax2 = fixed_aper * np.linspace(-1, 1, A)
    X2, Y2 = np.meshgrid(xax2, yax2)

    # X2 = X / period
    # Y2 = Y / aper
    eye = (abs(Y2 + spitz_pos*X2/np.pi) <= (np.cos(X2) ** 2))*1

    TG = conv2(pos_mat, eye, mode='same')
    return TG, pix_size


def zone_plate(lam=13.5e-9, f=.1, N=2000, pix=1e-6, center=(0, 0)):
    Nx = N
    Ny = N
    tx = segment_around(center[0], pix, Nx)
    ty = segment_around(center[1], pix, Ny)
    X, Y = np.meshgrid(tx, ty)
    R2 = X**2 + Y**2
    r2_max = np.max(R2)
    # print(r2_max.size)
    n_max = int(np.ceil(2*(-f+np.sqrt(f**2+r2_max))/lam))
    n = np.arange(n_max)  # np.reshape(, (1, 1, n_max))
    r2_n = n*lam*f+0.25*(n**2)*(lam**2)

    mask = np.sum(R2[:, :, np.newaxis] > r2_n, 2) % 2
    return mask

def segment_around(center, dx, N):
    return np.linspace(center-(N-1)*dx/2, center+(N-1)*dx/2, N)
 # def resolution_chart(N=2000):
 #    # This func creates NxN matrix, with resolution chart (series of vertical lines with changing widths)
 #    # In: N - Matrix size edge (if 0.5=> in the middle, if 0 all is 1, if 1 all is 0
 #    # example: from matplotlib import pyplot as plt
 #    #          from optics import knife_edge
 #    #          plt.imshow(knife_edge(100,0.2))
 #    #          plt.colorbar()
 #    #          plt.imshow(np.transpose(1-knife_edge(100,0.2)))  # To receive knife_edge in Y axis, '1' up and '0' down
 #
 #    mask = np.ones([N,N])
 #    index0 = 0
 #    for x in range (int( np.log(np.floor(N/6))/np.log(2) )+1):  # We plot 3 line pairs for each width
 #        width = 2**x
 #        mask1 = knife_edge()




