// The product name comes from /brand.json at the repo root (the single place to rename the product).
import brandJson from "../../../../brand.json";
import type { Brand } from "../api/types";

export const brand: Brand = brandJson as Brand;
