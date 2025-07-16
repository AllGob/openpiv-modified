from typing import Union, Optional
import pathlib
import numpy as np

def vec_on_matrx(vec=np.array,matr=np.array):
    
    if matr.shape[1]==vec.shape[0]:
    
        return(np.array([matr[0][0]*vec[0]+matr[0][1]*vec[1]+matr[0][2]*vec[2],matr[1][0]*vec[0]+matr[1][1]*vec[1]+matr[1][2]*vec[2],matr[2][0]*vec[0]+matr[2][1]*vec[1]+matr[2][2]*vec[2]]))



def to_readblearray(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    flags: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,

):
    """Making array with format [[x1 y1 u1 v1 flags1 mask1]...[xn yn un vn flagsn maskn]]

    Parameters
    ----------
 
    x : 2d np.ndarray
        a two dimensional array containing the x coordinates of the
        interrogation window centers, in pixels.

    y : 2d np.ndarray
        a two dimensional array containing the y coordinates of the
        interrogation window centers, in pixels.

    u : 2d np.ndarray
        a two dimensional array containing the u velocity components,
        in pixels/seconds.

    v : 2d np.ndarray
        a two dimensional array containing the v velocity components,
        in pixels/seconds.

    flags : 2d np.ndarray
        a two dimensional integers array where elements corresponding to
        vectors: 0 - valid, 1 - invalid (, 2 - interpolated)
        default: None, will create all valid 0

    mask: 2d np.ndarray boolean, marks the image masked regions (dynamic and/or static)
        default: None - will be all False



    """
    if isinstance(u, np.ma.MaskedArray):
        u = u.filled(0.)
        v = v.filled(0.)

    if mask is None:
        mask = np.zeros_like(u, dtype=int)

    if flags is None:
        flags = np.zeros_like(u, dtype=int)

    # build output array
    out = np.vstack([m.flatten() for m in [x, y, u, v, flags, mask]])

    # return array in normal form
    return(out.T)


def save_array(
    filename: Union[pathlib.Path, str],
    out: np.ndarray,
    fmt: str = "%.4e",
    delimiter: str = "\t",
) -> None:
    """Save from array to an ascii file.

    Parameters
    ----------
    filename : string
        the path of the file where to save the flow field

    out : np.ndarray


    fmt : string
        a format string. See documentation of numpy.savetxt
        for more details.

    delimiter : string
        character separating columns

    Examples
    --------

    openpiv.tools2.save_array('field_001.txt',out,  fmt='%6.3f',
                        delimiter='\t')

    """

    # save data to file.
    np.savetxt(
        filename,
        out,
        fmt=fmt,
        delimiter=delimiter,
        header="x"
        + delimiter
        + "y"
        + delimiter
        + "u"
        + delimiter
        + "v"
        + delimiter
        + "flags"
        + delimiter
        + "mask",
    )

def array_on_matrix(
    a:list,
    matrx:list
)->list:
    """ Multiplication of results array on transformation matrix .

    Parameters
    ----------
    a : np.array
        array after tools2.readble_array()

    matrx : np.array
            matrix transform (looks like:
                                [[a11, a12, a13],[a21,a22,a23],[0,0,1]])

    """
    i=0
    
    while i < len(a):
        a[i][2]=(vec_on_matrx(np.array([a[i][0]+a[i][2],a[i][1]+a[i][3],1]),matrx)-vec_on_matrx(np.array([a[i][0],a[i][1],1]),matrx))[0]
        a[i][3]=(vec_on_matrx(np.array([a[i][0]+a[i][2],a[i][1]+a[i][3],1]),matrx)-vec_on_matrx(np.array([a[i][0],a[i][1],1]),matrx))[1]
        a[i][0]=(vec_on_matrx(np.array([a[i][0],a[i][1],1]),matrx))[0]
        a[i][1]=(vec_on_matrx(np.array([a[i][0],a[i][1],1]),matrx))[1]
        i=i+1
    return(a)   
    

    
    
    
    
    
