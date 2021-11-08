import taichi as ti
import handy_shader_functions as hsf

ti.init(arch=ti.gpu)


res_x = 640
res_y = 360
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

iterations = 17
formuparam = 0.53
volsteps = 20
stepsize = 0.1

zoom = 0.800
tile = 0.850
speed = 0.010

brightness = 0.0015
darkmatter = 0.300
distfading = 0.730
saturation = 0.850


@ti.kernel
def paint(iTime: ti.f32):
    for i, j in pixels:
        uv = ti.Vector([float(i) / res_x, float(j) / res_y]) - 0.5
        uv[1] *= res_y / res_x
        dir = ti.Vector([uv[0] * zoom, uv[1] * zoom, 1.0])
        time = iTime * speed + 0.25

        a1 = 0.5
        a2 = 0.8
        rot1 = ti.Vector([[ti.cos(a1), ti.sin(a1)], [-ti.sin(a1), ti.cos(a1)]])
        rot2 = ti.Vector([[ti.cos(a2), ti.sin(a2)], [-ti.sin(a2), ti.cos(a2)]])
        dir_xz = ti.Vector([dir[0], dir[2]])
        dir_xz = rot1 @ dir_xz
        dir[0] = dir_xz[0]
        dir[2] = dir_xz[1]
        dir_xy = ti.Vector([dir[0], dir[1]])
        dir_xy = rot2 @ dir_xy
        dir[0] = dir_xy[0]
        dir[1] = dir_xy[1]

        from_ = ti.Vector([1.0, 0.5, 0.5])
        from_ += ti.Vector([time * 2, time, -2.0])
        from_xz = ti.Vector([from_[0], from_[2]])
        from_xz = rot1 @ from_xz
        from_[0] = from_xz[0]
        from_[2] = from_xz[1]

        from_xy = ti.Vector([from_[0], from_[1]])
        from_xy = rot2 @ from_xy
        from_[0] = from_xy[0]
        from_[1] = from_xy[1]

        s = 0.1
        fade = 1.0

        v = ti.Vector([0.0, 0.0, 0.0])

        for r in range(volsteps):
            p = from_ + s * dir * 0.5
            p = ti.abs(ti.Vector([tile, tile, tile]) - hsf.mod(p, ti.Vector([tile*2.0, tile*2.0, tile*2.0])))
            a = pa = 0.0
            for k in range(iterations):
                p = ti.abs(p) / p.dot(p) - formuparam
                a += ti.abs(p.norm() - pa)
                pa = p.norm()

            dm = ti.max(0.0, darkmatter - a * a * 0.001)
            a *= a * a
            if r > 6:
                fade *= 1.0 * dm
            v += fade
            v += ti.Vector([s, s*s, s**4]) * a * brightness * fade
            fade *= distfading
            s += stepsize
        v = hsf.mix(ti.Vector([v.norm(), v.norm(), v.norm()]), v, saturation)
        pixels[i, j] = v * 0.01


gui = ti.GUI("star", (res_x, res_y))
t = 0.0
while True:
    t += 0.02
    paint(t)
    gui.set_image(pixels)
    gui.show()

