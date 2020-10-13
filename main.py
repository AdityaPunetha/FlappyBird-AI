import pygame
import time
import os
import random
import neat

pygame.font.init()
pygame.init()

##
win = pygame.display.set_mode((500, 800))
birdimg = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird1.png"))),
           pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird2.png"))),
           pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird3.png")))]
pipeimg = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")))
baseimg = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))
bgimg = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))
STAT_FONT = pygame.font.SysFont("comicsans", 50)
END_FONT = pygame.font.SysFont("comicsans", 70)

gen = 0


##

class Bird:
    imgs = birdimg
    mxrot = 25
    rotvel = 20
    anitim = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tcount = 0
        self.vel = 0
        self.height = self.y
        self.icount = 0
        self.img = self.imgs[0]

    def jump(self):
        self.vel = -10.5
        self.tcount = 0
        self.height = self.y

    def move(self):
        self.tcount += 1
        d = self.vel * self.tcount + 1.5 * self.tcount ** 2
        if d >= 16:
            d = 16
        if d < 0:
            d -= 2
        self.y = self.y + d
        if d < 0 or self.y < self.height + 50:
            if self.tilt < self.mxrot:
                self.tilt = self.mxrot
            else:
                if self.tilt > -90:
                    self.tilt -= self.rotvel

    def draw(self, win):
        self.icount += 1
        if self.icount < self.anitim:
            self.img = self.imgs[0]
        elif self.icount > self.anitim * 2:
            self.img = self.imgs[1]
        elif self.icount > self.anitim * 3:
            self.img = self.imgs[2]
        elif self.icount > self.anitim * 4:
            self.img = self.imgs[1]
        elif self.icount == self.anitim * 4 + 1:
            self.img = self.imgs[0]
            self.icount = 0
        if self.tilt <= 80:
            self.img = self.imgs[1]
            self.icount = self.anitim * 2
        roti = pygame.transform.rotate(self.img, self.tilt)
        nrect = roti.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        win.blit(roti, nrect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)


##
class Pipe:
    GAP = 200
    VEL = 5

    def __init__(self, x):
        self.x = x
        self.height = 0

        # where the top and bottom of the pipe is
        self.top = 0
        self.bottom = 0

        self.PIPE_TOP = pygame.transform.flip(pipeimg, False, True)
        self.PIPE_BOTTOM = pipeimg
        self.passed = False

        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        if b_point or t_point:
            return True

        return False


##
class Base:
    VEL = 5
    WIDTH = baseimg.get_width()
    IMG = baseimg

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


##
def drawwindow(win, bird, pipes, base, score):
    win.blit(bgimg, (0, 0))
    for pipe in pipes:
        pipe.draw(win)
    text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(text, (500 - text.get_width() - 10, 10))
    text = STAT_FONT.render("Gen: " + str(gen), 1, (255, 255, 255))
    win.blit(text, (10, 10))
    base.draw(win)
    for birds in bird:
        birds.draw(win)

    pygame.display.update()


def main(genomes, config):
    global gen
    gen += 1
    nets = []
    ge = []
    birds = []
    for _, g in genomes:
        g.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        ge.append(g)
    base = Base(730)
    pipes = [Pipe(700)]
    win = pygame.display.set_mode((500, 800))
    clock = pygame.time.Clock()
    run = True
    score = 0
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1
        else:
            run = False
            break

        for x, bird in enumerate(birds):
            ge[x].fitness += 0.1
            bird.move()

            output = nets[x].activate(
                (bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))

            if output[0] > 0.5:
                bird.jump()

        base.move()
        rem = []
        add_pipe = False
        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    # ge[x] -= 1
                    nets.pop(x)
                    ge.pop(x)
                    birds.pop(x)
                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            pipe.move()
        if add_pipe:
            score += 1
            for g in ge:
                g.fitness += 5
            pipes.append(Pipe(700))
        for r in rem:
            pipes.remove(r)
        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
                nets.pop(x)
                ge.pop(x)
                birds.pop(x)
        drawwindow(win, birds, pipes, base, score)


def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    winner = p.run(main, 50)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
