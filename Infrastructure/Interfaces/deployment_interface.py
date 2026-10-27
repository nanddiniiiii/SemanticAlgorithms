"""Deployment contract reserved for NS-3, Sionna system, and O-RAN adapters."""

from abc import ABC, abstractmethod


class DeploymentInterface(ABC):
    @abstractmethod
    def deploy(self, plan: object) -> object:
        raise NotImplementedError
