{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "P = 1000\n",
    "N = 9\n",
    "M = 3\n",
    "s = 0.01\n",
    "m = torch.randn( N+1, M) # rand normal\n",
    "x = torch.rand( P, N+1) # rand unifom\n",
    "x[:,-1] = 1\n",
    "z = torch.mm(x, m) # matrix multiply\n",
    "w = torch.randn( N+1, M, requires_grad=True)# w va a tener difenciacion automatica\n",
    "xn = x + s*torch.randn( P, N+1) # agrego ruido"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "100 0.2704870910644531\n",
      "200 0.023847768783569336\n",
      "300 0.007805245876312256\n",
      "400 0.005105503559112549\n",
      "500 0.004267853260040283\n",
      "600 0.0039705228805542\n",
      "700 0.0038628830909729003\n",
      "800 0.003823808193206787\n",
      "900 0.0038096160888671875\n",
      "1000 0.0038044629096984865\n",
      "1100 0.0038025929927825927\n",
      "1200 0.0038019115924835204\n",
      "1300 0.0038016648292541503\n",
      "1400 0.003801575183868408\n",
      "1500 0.0038015429973602293\n",
      "1600 0.0038015315532684327\n",
      "1700 0.0038015263080596923\n",
      "1800 0.003801524877548218\n",
      "1900 0.0038015246391296387\n",
      "2000 0.003801524877548218\n"
     ]
    }
   ],
   "source": [
    "lr = 1e-4\n",
    "E, t = 1., 0\n",
    "while E > 1e-3 and t < 2000:\n",
    "    y = torch.mm(xn, w)\n",
    "    error = (y-z).pow(2).sum()\n",
    "    error.backward()\n",
    "    with torch.no_grad():\n",
    "        w -= lr*w.grad # grad es atributo\n",
    "        w.grad.zero_() # _ al final = inplace\n",
    "    E = error.item()/P # item del tensor, ya que para dimension 1 es aprox [Number]\n",
    "    t += 1\n",
    "    if t % 100 == 0:\n",
    "        print(t, E)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1.9707656974787824e-05\n"
     ]
    }
   ],
   "source": [
    "print( (torch.mm(x,w)-z).pow(2).mean().item() )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tensor([[-1.5254,  0.0392,  2.4099],\n",
      "        [-1.1489,  1.3082, -0.5209],\n",
      "        [ 1.3937, -0.2563,  1.5323],\n",
      "        [-0.7462, -0.2940, -0.9337],\n",
      "        [ 0.8068, -0.7880,  0.1776],\n",
      "        [ 1.9319, -0.8336,  1.0428],\n",
      "        [-1.5844, -0.8729, -0.2104],\n",
      "        [ 0.8362,  0.5250, -0.1616],\n",
      "        [-1.3070,  0.5117, -0.8191],\n",
      "        [-0.0690,  0.2749, -2.0617]])\n",
      "tensor([[-1.5269,  0.0370,  2.4051],\n",
      "        [-1.1500,  1.3058, -0.5232],\n",
      "        [ 1.3924, -0.2516,  1.5241],\n",
      "        [-0.7434, -0.2920, -0.9426],\n",
      "        [ 0.7993, -0.7807,  0.1625],\n",
      "        [ 1.9346, -0.8350,  1.0438],\n",
      "        [-1.5833, -0.8693, -0.2170],\n",
      "        [ 0.8270,  0.5256, -0.1645],\n",
      "        [-1.3081,  0.5118, -0.8264],\n",
      "        [-0.0622,  0.2684, -2.0350]], requires_grad=True)\n"
     ]
    }
   ],
   "source": [
    "print(m)\n",
    "print(w)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
