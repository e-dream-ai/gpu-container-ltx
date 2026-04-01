variable "DOCKERHUB_REPO" {
  default = "edreamai"
}

variable "DOCKERHUB_IMG" {
  default = "gpu-container-ltx"
}

variable "RELEASE_VERSION" {
  default = "latest"
}

group "default" {
  targets = ["ltx23"]
}

target "base" {
  context = "."
  dockerfile = "Dockerfile"
  target = "base"
  platforms = ["linux/amd64"]
  tags = ["${DOCKERHUB_REPO}/${DOCKERHUB_IMG}:${RELEASE_VERSION}-base"]
}

target "ltx23" {
  context = "."
  dockerfile = "Dockerfile"
  target = "final"
  tags = ["${DOCKERHUB_REPO}/${DOCKERHUB_IMG}:${RELEASE_VERSION}"]
  inherits = ["base"]
}
